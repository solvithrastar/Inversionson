import obspy


def select_component_from_stream(st: obspy.core.Stream, component: str):
    """
    Helper function selecting a component from a Stream an raising the proper
    error if not found.

    This is a bit more flexible then stream.select() as it works with single
    letter channels and lowercase channels.

    :param st: Obspy stream
    :type st: obspy.core.Stream
    :param component: Name of component of stream
    :type component: str
    """
    component = component.upper()
    component = [tr for tr in st if tr.stats.channel[-1].upper() == component]
    if not component:
        raise Exception(
            "Component %s not found in Stream." % component
        )
    elif len(component) > 1:
        raise Exception(
            "More than 1 Trace with component %s found "
            "in Stream." % component
        )
    return component[0]