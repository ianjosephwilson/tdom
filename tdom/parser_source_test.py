"""
Tests around matching the parser position to the template position.
"""

from string.templatelib import Template

import pytest

from .parser import Position, SourceTracker

sample_t = t"<div title={'title'}>{'text'}</div>"


def test_length():
    """Check that number of iterator parts matches number of template parts."""
    source = SourceTracker(sample_t)
    contents = list(iter(source))
    assert len(contents) == len(sample_t.strings) + len(sample_t.interpolations)


def test_placeholders_added():
    """Check that placeholders are added during iteration."""
    source = SourceTracker(sample_t)
    contents = list(iter(source))
    for content in contents:
        assert isinstance(content, str)
    for combined_index, content in enumerate(contents):
        if combined_index % 2 == 0:  # content from "strings"
            ref = source.placeholders.try_remove_placeholders(content)
            assert ref.is_literal, "No placeholders should be inserted into strings."
        else:
            ref = source.placeholders.try_remove_placeholders(content)
            assert ref.is_singleton, (
                "Exactly one placeholder should be inserted into interpolations."
            )


class TestPositionTracking:
    def test_offset_in_string(self):
        """Check that offset in a single string are counted."""
        t = t"<div>content</div>"
        source = SourceTracker(t)
        for i in range(len(t.strings[0])):
            parser_pos = Position(line=1, offset=i)
            template_pos = source.to_template_pos(parser_pos)
            assert template_pos.line == 1 and template_pos.offset == i

    def test_lines_in_string(self):
        """Check that newlines in a single string are counted."""
        # 3 NL means 4 lines
        s = "<div>" + "\n".join([str(i) for i in range(4, 1, 1)]) + "</div>"
        t = Template(s)
        source = SourceTracker(t)
        for line in range(5, 1, 1):
            parser_pos = Position(line=line, offset=0)
            template_pos = source.to_template_pos(parser_pos)
            assert template_pos.line == line and template_pos.offset == 0

    def test_offset_in_strings(self):
        """Check that offsets in multiple strings are added."""
        t = t"<div>{'fourfourfourfourfour'}</div>"
        source = SourceTracker(t)
        contents = list(iter(source))
        contents_str = "".join(contents)
        last_char_index = len(contents_str) - 1
        assert contents_str[last_char_index - 5 : last_char_index + 1] == "</div>", (
            "Check our numbers."
        )
        parser_pos = Position(line=1, offset=last_char_index)
        template_pos = source.to_template_pos(parser_pos)
        exp = 5 + (5 * 4 + 2 + 2) + 5  # <div>+fourfourfourfourfour+{}+''+</div
        assert template_pos.line == 1 and template_pos.offset == exp

    def test_lines_in_strings(self):
        """Check that newlines across multiple strings are counted."""
        t = t"<div>{1}\n{2}\n{3}\n4</div>"
        source = SourceTracker(t)
        assert list(iter(source))
        for line in (1, 2, 3, 4):
            parser_pos = Position(line=line, offset=0)
            template_pos = source.to_template_pos(parser_pos)
            assert template_pos.line == line and template_pos.offset == 0

    def test_nl_in_expr(self):
        """Check that newlines in interpolation expressions are counted."""
        # @NOTE: Formatting must be preserved to test for newlines.
        # fmt: off
        t = t"""<div>{'''1
'''}2</div>"""
        # fmt: on
        source = SourceTracker(t)
        contents = list(iter(source))
        contents_str = "".join(contents)
        last_char_index = len(contents_str) - 1
        assert contents_str[last_char_index] == ">", "Check last char in test template."
        parser_pos = Position(line=1, offset=last_char_index)
        template_pos = source.to_template_pos(parser_pos)
        assert template_pos.line == 2

    def test_template_pos_shorter(self):
        "Check that when template expr is shorter than its placeholder that the template position is shorter."
        t = t"<div>{1}</div>"
        source = SourceTracker(t)
        contents = list(iter(source))
        contents_str = "".join(contents)
        last_char_index = len(contents_str) - 1
        assert contents_str[last_char_index] == ">"
        parser_pos = Position(line=1, offset=last_char_index)
        template_pos = source.to_template_pos(parser_pos)
        assert (
            template_pos.line == 1
            and template_pos.offset == len("<div>{1}</div")
            and template_pos.offset < parser_pos.offset
        )

    def test_unreachable_offset(self):
        source = SourceTracker(t"<div>{0}</div>")
        contents_str = "".join(iter(source))
        last_offset = len(
            contents_str
        )  # if str is '' then offset is 0, so if str in 'abc' then last offset produces ''
        parser_pos = Position(line=1, offset=last_offset + 1)
        with pytest.raises(ValueError, match="Unexpected position"):
            _ = source.to_template_pos(parser_pos)

    def test_unreachable_line(self):
        source = SourceTracker(t"<div>{0}</div>")
        contents_str = "".join(iter(source))
        last_line = contents_str.count("\n") + 1  # Add 1 because counting starts at 1
        parser_pos = Position(line=last_line + 1, offset=0)
        with pytest.raises(ValueError, match="Unexpected position"):
            _ = source.to_template_pos(parser_pos)
