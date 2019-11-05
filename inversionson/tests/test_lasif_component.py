import pytest
import os
import lasif.api
import shutil

from inversionson.tests.testing_helpers import DummyProject

# Might need to implement this with opening and closing the project
# without using a tmp_path. Just use a real path

event_names = ["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14",
               "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20"]
@pytest.fixture(scope="module")
def pro():
    # Create a physical tmp folder
    # Create a project in there
    dir_path = os.path.dirname(os.path.realpath(__file__))
    project_path = os.path.join(dir_path, "tmp")
    os.makedirs(project_path)
    pro = DummyProject(project_path)
    yield pro
    # Here close everything down. Delete all folders.
    shutil.rmtree(project_path)

@pytest.mark.parametrize('event', [event_names[0], event_names[1]])
@pytest.mark.parametrize('attrib', ["latitude", "longitude", "mrr", "mtp",
                                    "stf_file", "dataset"])
def test_get_source(pro, event, attrib):

    source = pro.comm.lasif.get_source(event)

    if isinstance(source, list):
        source = source[0]

    assert attrib in source.keys()

@pytest.mark.parametrize('it_name', ["init", "bullshit", "ITERATION_init"])
def test_has_iteration(pro, it_name):
    # pro = DummyProject(tmp_path)

    lasif.api.set_up_iteration(pro.comm.lasif.lasif_root, "init")

    if "init" in it_name:
        assert pro.comm.lasif.has_iteration(it_name)
    else:
        assert not pro.comm.lasif.has_iteration(it_name)

    lasif.api.set_up_iteration(pro.comm.lasif.lasif_root,
                               iteration="init",
                               remove_dirs=True)


@pytest.mark.parametrize('events', [[], [event_names[0]]])
def test_set_up_iteration(pro, events):
    # pro = DummyProject(tmp_path)
    # lasif.api.set_up_iteration(pro.comm.lasif.lasif_root, "init", events)
    pro.comm.lasif.set_up_iteration("init", events)

    assert pro.comm.lasif.has_iteration("init")
    num_events = len(lasif.api.list_events(
                        pro.comm.lasif.lasif_root,
                        iteration="init",
                        output=True))
    if not events:
        assert num_events == 2
    else:
        assert num_events == 1
    lasif.api.set_up_iteration(pro.comm.lasif.lasif_root,
                               iteration="init",
                               remove_dirs=True)


@pytest.mark.parametrize('event', [event_names[0], event_names[1]])
def test_has_mesh(pro, event):
    # pro = DummyProject(tmp_path)
    has = pro.comm.lasif.has_mesh(event)
    if event == event_names[0]:
        assert has
    else:
        assert not has


def test_list_events(pro):
    # pro = DummyProject(tmp_path)
    events = pro.comm.lasif.list_events()

    assert set(events) == set(event_names)


def test_get_minibatch(pro):
    pro.comm.project.change_attribute(
        attribute="initial_batch_size",
        new_value=2
    )
    first = True
    batch = pro.comm.lasif.get_minibatch(first)

    assert set(batch) == set(event_names)

    pro.comm.project.change_attribute(
        attribute="initial_batch_size",
        new_value=1
    )
    batch = pro.comm.lasif.get_minibatch(first)
    assert len(batch) == 1


@pytest.mark.parametrize('event', [event_names[0], event_names[1]])
def test_move_mesh(pro, event, capsys):

    if event == event_names[1]:
        event_mesh = os.path.join(
            pro.comm.project.lasif_root,
            "MODELS",
            "EVENT_MESHES",
            event,
            "mesh.h5"
        )
        if not os.path.exists(os.path.dirname(event_mesh)):
            os.makedirs(os.path.dirname(event_mesh))
        pro.dummy_file(event_mesh)
    sim_mesh = os.path.join(
        pro.comm.project.lasif_root,
        "MODELS",
        "ITERATION_it0000_model",
        event,
        "mesh.h5"
    )
    if event == event_names[0]:
        assert os.path.exists(sim_mesh)
    else:
        assert not os.path.exists(sim_mesh)
    pro.comm.lasif.move_mesh(event, iteration="it0000_model")
    assert os.path.exists(sim_mesh)
    captured = capsys.readouterr()
    if event == event_names[0]:
        assert "correct path for iteration" in captured.out
    else:
        assert "has been moved to correct path for" in captured.out


def test_get_master_model(pro):
    # This test needs to be changed when the actual code is fixed
    model = os.path.join(
        pro.comm.project.lasif_root,
        "MODELS",
        "Globe3D_csem_100.h5"
    )
    assert pro.comm.lasif.get_master_model == model


