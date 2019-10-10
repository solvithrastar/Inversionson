import pytest
import os
import lasif.api

from inversionson.tests.testing_helpers import DummyProject

@pytest.mark.parametrize('it_name', ["init", "bullshit", "ITERATION_init"])
def test_has_iteration(tmp_path, it_name):
    pro = DummyProject(tmp_path)

    lasif.api.set_up_iteration(pro.comm.lasif.lasif_root, "init")

    if "init" in it_name:
        assert pro.comm.lasif.has_iteration(it_name)
    else:
        assert not pro.comm.lasif.has_iteration(it_name)


event_names = ["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
               "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-11"]

@pytest.mark.parametrize('events', [[], [event_names[0]]])
def test_set_up_iteration(tmp_path, events):
    pro = DummyProject(tmp_path)
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


@pytest.mark.parametrize('event', [event_names[0], event_names[1]])
def test_has_mesh(tmp_path, event):
    pro = DummyProject(tmp_path)
    has = pro.comm.lasif.has_mesh(event)
    if event == event_names[0]:
        assert has
    else:
        assert not has

