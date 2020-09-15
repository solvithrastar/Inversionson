import paramiko
from pathlib import Path
from typing import Tuple, Dict, List
import tqdm
import socket
import os
import inspect
import stat

CUT_SOURCE_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))),
    "remote_scripts",
    "cut_and_clip.py",
)

print(CUT_SOURCE_SCRIPT_PATH)


class retry():
    """
    Decorator that will keep retrying the operation after a timeout.
    """

    def __init__(self, retries: int):
        self.retries = retries

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            for _ in range(self.retries):
                try:
                    retval = f(*args, **kwargs)
                except socket.timeout:
                    continue
                else:
                    return retval
                raise

        return wrapped_f


def _tqdm_callback(*args, **kwargs):
    """
    Callback function to be used with the remote putter and getter.
    """
    pbar = tqdm.tqdm(*args, **kwargs)
    cache = [0]

    def update_bar(current, total):
        pbar.total = total
        pbar.update(current - cache[0])
        cache[0] = current

    return update_bar, pbar


class DaintClient():
    """ This class deals with everything on Daint.
    Always specify full paths

    Might be cool to add this as a component, but maybe it's not so stable
    """
    def __init__(self, hostname, username):
        self.hostname = hostname
        self.username = username
        self.keyring_settings = None
        self.verbose = True
        self._init_ssh_and_stfp_clients()

    def __del__(self):
        """
        Close the clients.
        """
        if hasattr(self, "ssh_client"):
            self.ssh_client.close()
        if hasattr(self, "sftp_client"):
            self.sftp_client.close()

    def _init_ssh_and_stfp_clients(self):
        """
        Initialize SSH and SFTP clients.
        """
        # Load the config.
        user_config_file = Path.home() / Path(".ssh") / Path("config")
        ssh_config = paramiko.SSHConfig()

        # Only parse config file if it exists.
        if user_config_file.exists():
            with open(user_config_file) as fh:
                ssh_config.parse(fh)

        # Use it to get host. info["hostname"] will always be set afterwards.
        info = ssh_config.lookup(self.hostname)

        # Overwrite user it with the one set in the config.
        if "user" in info:
            del info["user"]
        info["username"] = self.username

        self.ssh_client = paramiko.SSHClient()
        # Should be safe enough in our controlled environment.
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.load_system_host_keys()

        # Use the keyring library to encrypt the private SSH key if required.
        if self.keyring_settings:
            import keyring
            private_key_file = Path.home() / Path(".ssh") / Path("id_rsa")
            pw = keyring.get_password(*self.keyring_settings)
            if not pw:
                msg = (
                    "Failed to get SSH key password from keyring. Make sure "
                    "keyring.get_password(servicename, username) works from "
                    "within Python and that it returns the correct password "
                    "with the settings chosen in the salvus-flow config.")
                raise ValueError(msg)
            pkey = paramiko.RSAKey.from_private_key_file(
                filename=private_key_file,
                password=pw)
        else:
            pkey = None

        # Follow proxy commands if set.
        if "proxycommand" in info:
            sock = paramiko.ProxyCommand(info["proxycommand"])
        else:
            sock = None

        self.ssh_client.connect(
            username=info["username"],
            hostname=info["hostname"],
            pkey=pkey,
            sock=sock,
            # Two minutes should be good for most things.
            timeout=120)

        self.sftp_client = self.ssh_client.open_sftp()

        print("Connected to Daint.")

    def run_ssh_command(self, cmd: str, assert_ok: bool = True,
                        environment: Dict[str, str] = None) \
            -> Tuple[int, List[str], List[str]]:
        """
        Run the ssh command on the remote machine.

        :param cmd: Command to run.
        :param assert_ok: Assert that it exits with code zero, otherwise raise
            a remote execution error with more information.
        :param environment: A dict of shell environment variables to be merged
            into the default environment that the remote command executes with.
        :return:
        """
        # A bit hacky but the only reliable way I could find to inject
        # environment variables.
        if environment:
            prefix = [f"export {k}={v} && " for (k, v) in environment.items()]
            cmd = f"{' '.join(prefix)} {cmd}"
        if self.verbose:
            print("Executing command over SSH: '%s'" % cmd)
        _, stdout, stderr = self.ssh_client.exec_command(cmd)
        # Force synchronous execution.
        exit_status = stdout.channel.recv_exit_status()
        stdout = stdout.readlines()
        stderr = stderr.readlines()

        if assert_ok and exit_status != 0:
            nl = "\n"
            msg = f"Command '{cmd}' on {self.hostname} returned with exit " \
                  f"code {exit_status}. stderr: {nl}{nl.join(stderr)}"
            raise Exception(msg)

        return exit_status, stdout, stderr

    @retry(5)
    def remote_exists(self, path: str) -> bool:
        """
        Check if the remote path exists or not.
        """
        path = str(path)
        try:
            self.sftp_client.stat(path)
        except IOError as e:
            if e.args[0] == 2:
                return False
            raise
        else:
            return True

    @retry(5)
    def remote_put(self, localpath: Path, remotepath: Path,
                   progressbar: bool = False) -> None:
        if progressbar:
            callback, pbar = _tqdm_callback(
                unit="b", unit_scale=True, desc=remotepath.parts[-1])
        else:
            callback = None
        # For open files and bytes/io.
        if hasattr(localpath, "read") and hasattr(localpath, "seek"):
            return self.sftp_client.putfo(fl=localpath,
                                          remotepath=str(remotepath),
                                          callback=callback)
        # Directly copy files.
        else:
            return self.sftp_client.put(localpath=str(localpath),
                                        remotepath=str(remotepath),
                                        callback=callback)

    @retry(5)
    def remote_get(self, remotepath: Path, localpath: Path,
                   progressbar: bool = False) -> None:
        if progressbar:
            callback, pbar = _tqdm_callback(
                unit="b", unit_scale=True, desc=remotepath.parts[-1])
        else:
            callback = None
        self.sftp_client.get(remotepath=str(remotepath),
                             localpath=str(localpath),
                             callback=callback)
        if progressbar:
            # Trick tqdm to show something even for zero size files.
            if not pbar.total and not pbar.n:
                pbar.total = 1E-10
                pbar.n = 1E-10
            pbar.close()

    @retry(5)
    def remote_mkdir(self, path: Path, mode: int=511):
        return self.sftp_client.mkdir(path=str(path), mode=mode)

    @retry(5)
    def remote_rmdir(self, path: Path, quiet: bool=True):
        if not self.remote_exists(path=path):
            return
        for f in self.remote_listdir(path=path):
            f = Path(path) / f
            # Check if directory or file and recurse.
            if stat.S_ISDIR(self.sftp_client.stat(str(f)).st_mode):
                self.remote_rmdir(f, quiet=quiet)
            else:
                if not quiet:
                    print(f"🗑  Deleting file   {f} ...")
                self.sftp_client.unlink(str(f))
        if not quiet:
            print(f"🗑  Deleting folder {path} ...")
        self.sftp_client.rmdir(path=str(path))



