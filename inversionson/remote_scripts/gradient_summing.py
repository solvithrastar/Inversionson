import h5py
import sys
import toml
import shutil
import os

# Here I can add a scripts which adds the relevant fields to the mesh.
def sum_gradient(gradients: list, output_gradient: str,
                 parameters: list) -> bool:
    """
    Clip the gradient to remove abnormally high/low values from it.
    Discrete gradients sometimes have the problem of unphysically high
    values, especially at source/receiver locations so this should be
    taken care of by cutting out a region around these.

    :param gradients: List of paths to mesh containing gradient
    :type gradients: list
    :param output_gradient: path of output
    :type output_gradient: str
    :param parameters: Parameters to clip
    :type parameters: list
    :return bool to indicate that function ran succesfully till the end
    :rtype bool
    """
    first = True
    tmp_file = "temo_gradient_sum.h5"
    for grad_file in gradients:
        print(grad_file)

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
                .decode()
                .replace(" ", "")
                .split("|")
            )
            indices = []
            for param in parameters:
                indices.append(dim_labels.index(param))

            # go to next gradient
            first = False
            continue

        # open file, read_data, add to summed gradient and close.
        gradient = h5py.File(grad_file, "r+")
        data = gradient["MODEL/data"]
        for i in indices:
            summed_gradient_data[:, i, :] = summed_gradient_data[:, i, :] + \
                                            data[:, i, :],
        gradient.close()

    # finally close the summed_gradient
    summed_gradient.close()

    # This is done to ensure that the file is only there when the above
    # was successful.
    shutil.move(tmp_file, output_gradient)
    return True


if __name__ == "__main__":
    """
    Call with python name_of_script toml_filename
    """
    toml_filename = sys.argv[1]
    info = toml.load(toml_filename)
    gradient_filenames = info["filenames"]
    parameters = info["parameters"]
    output_gradient = info["output_gradient"]

    print("Remote summing of gradients started...")

    if sum_gradient(gradient_filenames, output_gradient, parameters):
        # I could add something here, to ensure that it ran successfully
        print("Seems to have worked!")


    # Set referece frame to spherical
    print("Set reference frame")
    with h5py.File(output_gradient, "r+") as f:
        attributes = f["MODEL"].attrs
        attributes.modify("reference_frame", b"spherical")

    with open(toml_filename, "w") as fh:
        toml.dump(info, fh)
