import h5py
import sys
import toml
import shutil
import numpy as np
import os


def sum_gradient(
    gradients: list, output_gradient: str, parameters: list, batch_average=False
) -> list:
    """
    Sum a list of gradients. This function is called on the remote cluster,
    and expects to be able to use h5py and Python.

    :param gradients: List of paths to mesh containing gradient
    :type gradients: list
    :param output_gradient: path of output
    :type output_gradient: str
    :param parameters: Parameters to clip
    :type parameters: list
    :param batch_average: Set to try to divide the sum by
    the number of gradients. Useful for stochastic methods.
    :type batch_average: bool
    :return gradient_norms: list of gradient norms
    :rtype list
    """
    first = True
    tmp_file = "temo_gradient_sum.h5"
    gradient_norms = []
    for grad_file in gradients:
        if first:
            # copy to target destination in which we will sum
            shutil.copy(grad_file, tmp_file)
            summed_gradient = h5py.File(tmp_file, "r+")
            summed_gradient_data = summed_gradient["MODEL/data"]
            # Get dimension indices of relevant parameters
            # These should be constant for all gradients, so this is only done
            # once.
            dim_labels = summed_gradient_data.attrs.get("DIMENSION_LABELS")[1][1:-1]
            if type(dim_labels) != str:
                dim_labels = dim_labels.decode()
            dim_labels = dim_labels.replace(" ", "").split("|")
            indices = [dim_labels.index(param) for param in parameters]
            sorted_indices = sorted(indices)
            # go to next gradient
            first = False
            summed_gradient_data_copy = summed_gradient_data[:, :, :].copy()
            grad_dat = summed_gradient_data_copy[:, sorted_indices, :]
            gradient_norms.append(np.sqrt(np.sum(grad_dat**2)))
            continue
        # This assumes the indices remain the same.
        # open file, read_data, add to summed gradient and close.
        gradient = h5py.File(grad_file, "r+")
        data = gradient["MODEL/data"]
        grad_dat = data[:, sorted_indices, :].copy()
        gradient_norms.append(np.sqrt(np.sum(grad_dat**2)))
        for i in indices:
            summed_gradient_data_copy[:, i, :] = (
                data[:, i, :] + summed_gradient_data_copy[:, i, :]
            )

        gradient.close()

    # divide by the number of gradients to obtain a batch average
    if batch_average:
        # only average the actual model parameters.
        for i in indices:
            summed_gradient_data_copy[:, i, :] /= len(gradients)
    # finally close the summed_gradient
    summed_gradient_data[:, :, :] = summed_gradient_data_copy[:, :, :]
    summed_gradient.close()

    # This is done to ensure that the file is only there when the above
    # was successful.
    shutil.move(tmp_file, output_gradient)
    return gradient_norms


if __name__ == "__main__":
    """
    Script to perform gradient summing. Also writes a file with the norms
    of each individual gradient.
    Call script with python name_of_script info_filename

    the info toml should contain a dict with keys:
    -filenames
    -output_gradient
    -events_list
    -parameters
    -gradient_norms_path
    -batch_average

    Events list and filenames should have the same ordering as the files
    that are summed.
    """
    toml_filename = sys.argv[1]
    info = toml.load(toml_filename)

    if os.path.exists(info["output_gradient"]):
        os.remove(info["output_gradient"])

    gradient_norms = sum_gradient(
        info["filenames"],
        info["output_gradient"],
        info["parameters"],
        info["batch_average"],
    )

    gradient_norm_dict = {
        info["events_list"][i]: gradient_norms[i]
        for i in range(len(info["events_list"]))
    }
    if info["gradient_norms_path"]:
        with open(info["gradient_norms_path"], "w") as fh:
            toml.dump(gradient_norm_dict, fh)

    # Set reference frame to spherical. This is needed for smoothing later
    with h5py.File(info["output_gradient"], "r+") as f:
        attributes = f["MODEL"].attrs
        attributes.modify("reference_frame", b"spherical")
