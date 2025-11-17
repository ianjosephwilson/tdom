from string.templatelib import Interpolation

from markupsafe import Markup

from .utils import format_interpolation as base_format_interpolation


def _format_safe(value: object, format_spec: str) -> str:
    """Use Markup() to mark a value as safe HTML."""
    assert format_spec == "safe"
    return Markup(value)


def _format_unsafe(value: object, format_spec: str) -> str:
    """Convert a value to a plain string, forcing it to be treated as unsafe."""
    assert format_spec == "unsafe"
    return str(value)


CUSTOM_FORMATTERS = (("safe", _format_safe), ("unsafe", _format_unsafe))


def format_interpolation(interpolation: Interpolation) -> object:
    return base_format_interpolation(
        interpolation,
        formatters=CUSTOM_FORMATTERS,
    )
