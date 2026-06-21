from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class FrozenPosition:
    "A immutable position in a block of source code."

    line: int = 1
    " Line of code, starts at 1. "
    offset: int = 0
    " Offset from the start of the line, starts at 0. "


@dataclass(slots=True)
class Position:
    "A position in a block of source code."

    line: int = 1
    " Line of code, starts at 1. "
    offset: int = 0
    " Offset from the start of the line, starts at 0. "

    def freeze(self) -> FrozenPosition:
        return FrozenPosition(line=self.line, offset=self.offset)


type HTMLAttribute = tuple[str, str | None]


@dataclass(frozen=True, slots=True)
class StartTagSourceInfo:
    "Retain the start tag information of the source."

    starttag_text: str
    " Entire starttag as parsed, includes placeholders, . "
    raw_attrs: tuple[HTMLAttribute, ...]
    " Attrs as parsed, includes placeholders. "
    startend: bool
    " Was parsed as startend tag, ie. <tag />. "
    pos: FrozenPosition
    " Position of the parser when the element starttag was parsed. "


@dataclass(frozen=True, slots=True)
class EndTagSourceInfo:
    """Record of the end tag information of the source."""

    pos: FrozenPosition
    " Position of the parser when the element endtag was parsed. "
