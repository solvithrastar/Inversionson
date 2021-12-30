import h5py
import sys
import toml
import shutil
import numpy as np
import os


def sum_gradient(gradients: list, output_gradient: str,
                 parameters: list, batch_average=False) -> bool:
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
    :return bool to indicate that function ran succesfully till the end
    :rtype bool
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
            dim_labels = (
                summed_gradient_data.attrs.get("DIMENSION_LABELS")[1][1:-1]
            )
            if not type(dim_labels) == str:
                dim_labels = dim_labels.decode()
            dim_labels = dim_labels.replace(" ", "").split("|")
            indices = []
            for param in parameters:
                indices.append(dim_labels.index(param))

            sorted_indices = indices.copy()
            sorted_indices.sort()
            # go to next gradient
            first = False
            summed_gradient_data_copy = summed_gradient_data[:, :, :].copy()
            grad_dat = summed_gradient_data_copy[:, sorted_indices, :]
            gradient_norms.append(np.sqrt(np.sum(grad_dat ** 2)))
            continue

        # open file, read_data, add to summed gradient and close.
        gradient = h5py.File(grad_file, "r+")
        data = gradient["MODEL/data"]
        grad_dat = data[:, sorted_indices, :].copy()
        gradient_norms.append(np.sqrt(np.sum(grad_dat ** 2)))
        for i in indices:
            summed_gradient_data_copy[:, i, :] = \
                data[:, i, :] + summed_gradient_data_copy[:, i, :]

        gradient.close()

    # divide by the number of gradients to obtain a batch average
    if batch_average:
        summed_gradient_data_copy /= len(gradients)
    # finally close the summed_gradient
    summed_gradient_data[:, :, :] = summed_gradient_data_copy[:, :, :]
    summed_gradient.close()

    # This is done to ensure that the file is only there when the above
    # was successful.
    shutil.move(tmp_file, output_gradient)
    return gradient_norms


if __name__ == "__main__":
    """
    Call with python name_of_script toml_filename
    """
    toml_filename = sys.argv[1]
    info = toml.load(toml_filename)
    gradient_filenames = info["filenames"]
    parameters = info["parameters"]
    output_gradient = info["output_gradient"]
    event_list = info["event_list"]
    if "batch_average" in info.keys():
        batch_average = info["batch_average"]
    else:
        batch_average = False

    norms_path = info["gradient_norms_path"]
    print("Remote summing of gradients started...")

    # clear the temporary file to avoid accidentally mixing up summed
    # gradients from prior iterations.
    if os.path.exists(output_gradient):
        os.remove(output_gradient)

    gradient_norms = sum_gradient(gradient_filenames, output_gradient,
                                  parameters, batch_average)

    gradient_norm_dict = {}
    for i in range(len(event_list)):
        gradient_norm_dict[event_list[i]] = gradient_norms[i]

    with open(norms_path, "w") as fh:
        toml.dump(gradient_norm_dict, fh)

    # I could add something here, to ensure that it ran successfully
    print("Seems to have worked!")
    # Set reference frame to spherical
    print("Set reference frame")
    with h5py.File(output_gradient, "r+") as f:
        attributes = f["MODEL"].attrs
        attributes.modify("reference_frame", b"spherical")

    # not sure if this is doing anything useful
    with open(toml_filename, "w") as fh:
        toml.dump(info, fh)
