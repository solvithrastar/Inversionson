import toml

from inversionson.remote_helpers.daint import DaintClient
from inversionson.remote_helpers.daint import CUT_SOURCE_SCRIPT_PATH

import os


def preprocess_remote_gradient(comm, gradient_path: str, event: str):
    """
    Cut sources and receivers from gradient before smoothing.
    We also clip the gradient to some percentile
    This can all be configured in information toml.

    :param comm inversionson communicator
    :param gradient_path: gradient path on remote
    :type gradient_path: str
    :param event: name of the event
    """

    # Connect to daint
    hostname = "daint"
    username = "dpvanher"
    daint = DaintClient(hostname, username)

    remote_inversionson_dir = os.path.join("/users", username, "Inversionson")
    print(remote_inversionson_dir)
    if not daint.remote_exists(remote_inversionson_dir):
        daint.remote_mkdir(remote_inversionson_dir)

    # copy processing script to daint
    remote_script = os.path.join(remote_inversionson_dir, "cut_and_clip.py")
    daint.remote_put(CUT_SOURCE_SCRIPT_PATH, remote_script)

    info = {}
    info["filename"] = str(gradient_path)
    info["cutout_radius_in_km"] = comm.project.cut_source_radius
    info["source_location"] = comm.lasif.get_source(event_name=event)

    info["clipping_percentile"] = comm.project.clip_gradient
    info["parameters"] = comm.project.inversion_params

    toml_filename = f"{event}_gradient_process.toml"
    with open(toml_filename, "w") as fh:
        toml.dump(info, fh)

    # put toml on daint
    remote_toml = os.path.join(remote_inversionson_dir, toml_filename)
    daint.remote_put(toml_filename, remote_toml)

    # Call script
    print("Calling script on Daint")
    print(daint.run_ssh_command(f"python {remote_script} {remote_toml}"))
