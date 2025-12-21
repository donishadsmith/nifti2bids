"""Helper functions."""


def iterable_to_str(str_list: list[str]) -> None:
    """Converts an iterable containing strings to strings."""
    return ", ".join(["'{a}'".format(a=x) for x in str_list])
