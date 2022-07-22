import subprocess as sp
import errno, os, sys
from pathlib import Path

def spacify(what):
    return " ".join(what)

class Git:
    def __init__(self, directory: str, author: str = None):
        self.dir = Path(directory)
        self.author = author

    def run(self, *args, check=True, capture_output=True, return_output=True):
        try:
            fp = sp.run(["git", "-C", self.dir] + list(args), check=check, capture_output=capture_output, text=True)
        except sp.CalledProcessError as e:
            print(e.stderr, file=sys.stderr)
            print(e.stdout)
            raise e

        if return_output:
            return fp.stdout
        else:
            return fp

    def add(self, *what):
        self.run("add", *what)

    def add_commit(self, *what, message = None):
        self.add(*what)
        if not message:
            message = "add " + spacify(what)
        return self.commit(message)

    def branch(self, branch):
        if self.is_branch(branch):
            self.run("switch", branch)
        else:
            self.run("switch", "-c", branch)

    def commit(self, msg):
        if self.author:
            cp = self.run("commit", "-m", msg, "--author", self.author, check=False, return_output=False)
        else:
            cp = self.run("commit", "-m", msg, check=False, return_output=False)
        # we don't throw an error if there is just nothing to commit
        if cp.returncode and "nothing to commit" not in cp.stdout:
            raise sp.CalledProcessError(cp.returncode, cp.args, output=cp.stdout, stderr=cp.stderr)
        return cp.stdout

    def ignore(self, *what):
        path = self.dir / ".gitignore"
        with open(path, "a") as f:
            for l in what:
                f.write(l + '\n')
        return self.add_commit(str(path), message = ".gitignore: " + spacify(what))

    def init(self):
        return self.run("init", "--initial-branch", "main") + self.run("lfs", "install", "--force", "--local")

    def is_branch(self, name):
        cp = self.run("show-ref", "-q", "--heads", name, check=False, return_output=False)
        return not cp.returncode

    def lfs_track(self, *what):
        f = ".gitattributes"
        for w in what:
            self.run("lfs", "track", w)
        if self.status(f):
            return self.add_commit(f, message = f"{f}: {spacify(what)}")
        return None

    def status(self, *what):
        # git status does not check existence
        for f in what:
            f = Path(f)
            if not self.dir in f.parents:
                f = self.dir / f
            if not f.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(f))

        output = self.run("status", "-z", "--ignored", *what).split()
        if output:
            return output[0]
        else:
            return None
