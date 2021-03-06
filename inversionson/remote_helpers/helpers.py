import toml
import os
import inspect
from inversionson import InversionsonError
from salvus.flow.api import get_site

CUT_SOURCE_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))),
    "remote_scripts",
    "cut_and_clip.py",
)

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
    daint = get_site(comm.project.site_name)
    username = daint.config["ssh_settings"]["username"]

    remote_inversionson_dir = os.path.join("/scratch/snx3000", username,
                                           "smoothing_info")

    if not daint.remote_exists(remote_inversionson_dir):
        daint.remote_mkdir(remote_inversionson_dir)

    # copy processing script to daint
    remote_script = os.path.join(remote_inversionson_dir, "cut_and_clip.py")
    if not daint.remote_exists(remote_script):
        daint.remote_put(CUT_SOURCE_SCRIPT_PATH, remote_script)

    if comm.project.cut_receiver_radius > 0.0:
        raise InversionsonError("Remote receiver cutting not implemented yet.")
    
    info = {}
    info["filename"] = str(gradient_path)
    info["cutout_radius_in_km"] = comm.project.cut_source_radius
    info["source_location"] = comm.lasif.get_source(event_name=event)

    info["clipping_percentile"] = comm.project.clip_gradient
    info["parameters"] = comm.project.inversion_params

    toml_filename = f"{event}_gradient_process.toml"
    with open(toml_filename, "w") as fh:
        toml.dump(info, fh)

    # put toml on daint and remove local toml
    remote_toml = os.path.join(remote_inversionson_dir, toml_filename)
    daint.remote_put(toml_filename, remote_toml)
    os.remove(toml_filename)

    # Call script
    print(daint.run_ssh_command(f"python {remote_script} {remote_toml}"))
