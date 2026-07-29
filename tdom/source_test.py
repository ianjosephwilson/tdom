from string.templatelib import Template

import pytest

from .source import LinePosition, SourceReader, template_repr
from .template_utils import PartPosition


class TestTemplateRepresentation:
    # whitespace is part of test
    # fmt: off
    @pytest.mark.parametrize(
        ("t", "result"),
        (
            (t"<div>{15!s:formatspec}</div>", "<div>{15!s:formatspec}</div>"),
            (t"<div>{15:formatspec}</div>", "<div>{15:formatspec}</div>"),
            (t"<div>{15!s}</div>", "<div>{15!s}</div>"),
            (t"<div>{15}</div>", "<div>{15}</div>"),
            (t"{15}", "{15}"),
            (t"", ""),
            (t"A{0}B{1}{2}C", "A{0}B{1}{2}C"),
            (t"ABC", "ABC"),
            (t"""<div>
</div>""", """<div>\n</div>"""),
            (t"""{'''
'''}""", """{'''\n'''}"""),
        )
    )
    def test_repr(self, t: Template, result: str):
        assert template_repr(t) == result
    # fmt: on


class TestToTemplatePosition:
    def test_origin(self):
        t = t"<div>{'content'}</div>"
        reader = SourceReader(template=t)
        source_pos = PartPosition(index=0, offset=0)
        assert reader.to_template_pos(source_pos) == LinePosition(line=1, offset=0)

    def test_offset_no_lines(self):
        t = t"<div>{'content'}</div>"
        reader = SourceReader(template=t)
        source_pos = PartPosition(index=1, offset=0)
        assert reader.to_template_pos(source_pos) == LinePosition(
            line=1, offset=len(t.strings[0])
        )

    def test_offset_full_interpolation(self):
        t = t"<div>{''!s:lower}</div>"  # conversion and formatspec
        reader = SourceReader(template=t)
        source_pos = PartPosition(index=2, offset=0)
        assert reader.to_template_pos(source_pos) == LinePosition(
            line=1, offset=len('<div>{""!s:lower}')
        )

    def test_line(self):
        # whitespace is part of test
        # fmt: off
        t = t"""<div>
{"content"}</div>"""
        # fmt: on
        reader = SourceReader(template=t)
        source_pos = PartPosition(index=2, offset=0)
        assert reader.to_template_pos(source_pos) == LinePosition(
            line=2, offset=len('{"content"}')
        )

    def test_line_in_interpolation(self):
        # whitespace is part of test
        # fmt: off
        t = t"""<div>
{'''
content
'''}</div>"""
        # fmt: on
        reader = SourceReader(template=t)
        source_pos = PartPosition(index=2, offset=0)
        assert reader.to_template_pos(source_pos) == LinePosition(
            line=4, offset=len("'''}")
        )
