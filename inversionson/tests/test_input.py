"""
Start experimenting with some tests.
This will not be tracked by git to begin with as it is only experimental
"""

import pytest
import os

from inversionson.tests.testing_helpers import DummyProject
from inversionson import InversionsonError


def test_initialization(tmp_path):
    """
    Ertu hress?
    """
    pro = DummyProject(tmp_path)
    assert pro.comm.project.inversion_root == pro.root_folder
    assert os.path.exists(os.path.join(
        pro.comm.project.inversion_root,
        "DOCUMENTATION"
    ))
    assert os.path.exists(os.path.join(
        pro.comm.project.inversion_root,
        "DOCUMENTATION",
        "ITERATIONS"
    ))
    assert pro.comm.project.paths["documentation"] == \
        os.path.join(pro.root_folder, "DOCUMENTATION")
    assert pro.comm.project.paths["iteration_tomls"] == \
        os.path.join(pro.root_folder, "DOCUMENTATION", "ITERATIONS")
    assert pro.comm.project.paths["salvus_opt"] == \
        os.path.join(pro.root_folder, "SALVUS_OPT")
    assert pro.comm.lasif.lasif_root == os.path.join(
        pro.root_folder, "LASIF_PROJECT")
    assert pro.comm.project.lasif_root == pro.comm.lasif.lasif_root
    assert pro.comm.project.modelling_params == ["RHO", "VP", "VS"]


def test_arrange_params(tmp_path):
    pro = DummyProject(tmp_path)

    params = ["VS", "VP"]
    assert pro.comm.project.arrange_params(params) == ["VP", "VS"]
    params = ["VP", "VS"]
    assert pro.comm.project.arrange_params(params) == ["VP", "VS"]
    params = ["VP", "VS", "RHO"]
    assert pro.comm.project.arrange_params(params) == ["RHO", "VP", "VS"]
    params = ["VPV", "VSV", "VPH", "VSH", "RHO", "QKAPPA", "QMU", "ETA"]
    assert pro.comm.project.arrange_params(params) == \
        ["VPV", "VPH", "VSV", "VSH", "RHO", "QKAPPA", "QMU", "ETA"]
    params = ["VPV", "VSV", "VPH", "VSH", "RHO"]
    assert pro.comm.project.arrange_params(params) == \
        ["VPV", "VPH", "VSV", "VSH", "RHO"]
    params = ["RHO", "VP", "VS", "QKAPPA", "QMU"]
    assert pro.comm.project.arrange_params(params) == \
        ["QKAPPA", "QMU", "RHO", "VP", "VS"]
    with pytest.raises(InversionsonError):
        pro.comm.project.arrange_params(["VP", "RHO", "MAG", "QKAPPA"])
