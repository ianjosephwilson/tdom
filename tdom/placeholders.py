from dataclasses import dataclass
import re
import random
import string
from string.templatelib import Interpolation
import typing as t

from .formatting import format_interpolation


_PLACEHOLDER_PREFIX = f"t🐍{''.join(random.choices(string.ascii_lowercase, k=2))}-"
_PLACEHOLDER_SUFFIX = f"-{''.join(random.choices(string.ascii_lowercase, k=2))}🐍t"
_PLACEHOLDER_PATTERN = re.compile(
    re.escape(_PLACEHOLDER_PREFIX) + r"(\d+)" + re.escape(_PLACEHOLDER_SUFFIX)
)


def _placeholder(i: int) -> str:
    """Generate a placeholder for the i-th interpolation."""
    return f"{_PLACEHOLDER_PREFIX}{i}{_PLACEHOLDER_SUFFIX}"


@dataclass(frozen=True, slots=True)
class _PlaceholderMatch:
    start: int
    end: int
    index: int | None


def _find_placeholder(s: str) -> int | None:
    """
    If the string is exactly one placeholder, return its index. Otherwise, None.
    """
    match = _PLACEHOLDER_PATTERN.fullmatch(s)
    return int(match.group(1)) if match else None


def _find_all_placeholders(s: str) -> t.Iterable[_PlaceholderMatch]:
    """
    Find all placeholders in a string, returning their positions and indices.

    If there is non-placeholder text in the string, its position is also
    returned with index None.
    """
    matches = list(_PLACEHOLDER_PATTERN.finditer(s))
    last_end = 0
    for match in matches:
        if match.start() > last_end:
            yield _PlaceholderMatch(last_end, match.start(), None)
        index = int(match.group(1))
        yield _PlaceholderMatch(match.start(), match.end(), index)
        last_end = match.end()
    if last_end < len(s):
        yield _PlaceholderMatch(last_end, len(s), None)


def _replace_placeholders(
    value: str, interpolations: tuple[Interpolation, ...]
) -> tuple[bool, object]:
    """
    Replace any placeholders embedded within a string attribute value.

    If there are no placeholders, return False and the original string.

    If there is exactly one placeholder and nothing else, return True and the
    corresponding interpolation value.

    If there are multiple placeholders or surrounding text, return True and
    a concatenated string with all placeholders replaced and interpolations
    formatted and converted to strings.
    """
    matches = tuple(_find_all_placeholders(value))

    # Case 1: No placeholders found
    if len(matches) == 1 and matches[0].index is None:
        return False, value

    # Case 2: Single placeholder and no surrounding text
    if len(matches) == 1 and matches[0].index is not None:
        index = matches[0].index
        formatted = format_interpolation(interpolations[index])
        return True, formatted

    # Case 3: Multiple placeholders or surrounding text
    parts = [
        value[match.start : match.end]
        if match.index is None
        else str(format_interpolation(interpolations[match.index]))
        for match in matches
    ]
    return True, "".join(parts)
