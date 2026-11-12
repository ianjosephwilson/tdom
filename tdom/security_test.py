"""
Security specific test suite.

This doesn't exhaust every permuation possible but tries to make sure any
sort of internal validation checks have not been disabled.

### Approach

- Maintain independent constants/values from library code to prevent failures
in the library code from bypassing these checks.
- Output should be matched exactly and negations should be avoided.  Otherwise
a different invalid version could *pass* but still be invalid.
"""

from string.templatelib import Template
from types import SimpleNamespace

import pytest

from .htmlspec import make_char_codepoint_reprs
from .parser import AttributeParsingError, ParsingError
from .processor import (
    AttributeProcessingError,
    ProcessContext,
    TemplateParserProxy,
    _make_default_template_processor,
)

processor_api = _make_default_template_processor(
    parser_api=TemplateParserProxy(),  # do not use cache
)


def make_ctx(**kwargs):
    return ProcessContext(**kwargs)


@pytest.fixture(scope="session")
def html():
    def _html(template: Template, assume_ctx: ProcessContext | None = None) -> str:
        if assume_ctx is None:
            assume_ctx = ProcessContext()
        return processor_api.process(template, assume_ctx=assume_ctx)

    return _html


TAB="\t"
LF="\u000a"  # \n
FF="\u000c"  # \f
CR="\u000d"  # \r
GT=">"
SOLIDUS="\u002f"  # /
LT = "<"
SQ = "'"
DQ = '"'
BT = "`"
AMP = "&"
NULL="\u0000"
EQ = "="

ENT_BY_CHAR = {
    LT: "&lt;",
    GT: "&gt;",
    SQ: "&#39;",
    DQ: "&#34;",
    AMP: "&amp;",
}

HEX_LT = r"\x3c"
assert HEX_LT != LT, "Confirm we didn't regress on escaping the \."


class TestDynamicAttributeName:
    """Test dynamic attribute names generated in the processor."""

    @pytest.mark.parametrize("invalid_char", (NULL, SOLIDUS, EQ, GT, DQ, SQ))
    def test_spread_attr(self, html, invalid_char):
        attrs = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div {attrs} />")

    @pytest.mark.parametrize("invalid_char", (NULL, SOLIDUS, EQ, GT, DQ, SQ))
    def test_aria_attr(self, html, invalid_char):
        aria = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            assert html(t"<div aria={aria} />")

    @pytest.mark.parametrize("invalid_char", (NULL, SOLIDUS, EQ, GT, DQ, SQ))
    def test_data_attr(self, html, invalid_char):
        data = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div data={data} />")


class TestLiteralAttributeName:
    """Test literal attribute names parsed in the parser."""

    @pytest.mark.parametrize("invalid_char", (NULL, EQ, DQ, SQ))
    def test_literal_attr(self, html, invalid_char):
        """Test that invalid chars are caught using a few specific examples."""
        with pytest.raises(
            AttributeParsingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            # We use `Template` here because we are testing the parsing of static text.
            _ = html(Template(f"<div {invalid_char}=1 />"))


class TestLiteralAttributeValue:
    """Test literal attribute values parsed in the parser."""

    @pytest.mark.skip(
        "HTMLParser chokes on dangling quotes and keeps waiting for more data."
    )
    def test_unquoted_double_quote(self, html):
        assert (
            html(t'<div x="></div>') == ""
        )  # This is currently not working, but not a security issue.

    def test_quoted_dangling_double_quote(self, html):
        assert html(t"""<div x='"'></div>""") == f'<div x="{ENT_BY_CHAR[DQ]}"></div>'

    @pytest.mark.skip(
        "HTMLParser chokes on dangling quotes and keeps waiting for more data."
    )
    def test_dangling_double_quote(self, html):
        assert html(t'<div x="></div>') == """<div x='"'></div>"""  # or escaped

    @pytest.mark.skip(
        "HTMLParser chokes on dangling quotes and keeps waiting for more data."
    )
    def test_unquoted_single_quote(self, html):
        assert (
            html(t"<div x='></div>") == ""
        )  # This is currently not working, but not a security issue.

    def test_quoted_single_quote(self, html):
        assert html(t"""<div x="'"></div>""") == f'<div x="{ENT_BY_CHAR[SQ]}"></div>'

    @pytest.mark.skip(
        "HTMLParser chokes on dangling quotes and keeps waiting for more data."
    )
    def test_dangling_single_quote(self, html):
        assert html(t"<div x='></div>") == """<div x="'"></div>"""  # or escaped

    @pytest.mark.parametrize("quoting", ("", "'", '"'))
    def test_backtick(self, html, quoting):
        assert (
            html(Template(f"<div x={quoting}`{quoting}></div>")) == '<div x="`"></div>'
        )

    @pytest.mark.parametrize("quoting", ("", "'", '"'))
    def test_amperstand(self, html, quoting):
        """
        This isn't restricted but could be abused.
        """
        assert (
            html(Template(f"<div x={quoting}&{quoting}></div>"))
            == f'<div x="{ENT_BY_CHAR[AMP]}"></div>'
        )

    @pytest.mark.parametrize("quoting", ("", "'", '"'))
    def test_equalsign(self, html, quoting):
        assert (
            html(Template(f"<div x={quoting}={quoting}></div>")) == '<div x="="></div>'
        )

    @pytest.mark.parametrize("quoting", ("", "'", '"'))
    def test_lessthan(self, html, quoting):
        assert (
            html(Template(f"<div x={quoting}<{quoting}></div>"))
            == f'<div x="{ENT_BY_CHAR[LT]}"></div>'
        )

    @pytest.mark.parametrize(
        "quoting", ("'", '"')
    )  # @NOTE: We cannot test > without quotes because the parsing does not work.
    def test_greaterthan(self, html, quoting):
        assert (
            html(Template(f"<div x={quoting}>{quoting}></div>"))
            == f'<div x="{ENT_BY_CHAR[GT]}"></div>'
        )

    def test_unquoted_greaterthan(self, html):
        """The parser should interpret the first > as the end of the tag so the following > is treated as text content and escaped."""
        assert html(Template("<div x=>></div>")) == f'<div x="">{ENT_BY_CHAR[GT]}</div>'


class TestDynamicAttributeValue:
    """Test dynamic attribute values generated in the processor."""

    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            (EQ, EQ),
            (BT, BT),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
           (SQ, ENT_BY_CHAR[SQ]),
           (DQ, ENT_BY_CHAR[DQ]),
           (AMP, ENT_BY_CHAR[AMP]),
           (GT, ENT_BY_CHAR[GT]),
           (LT, ENT_BY_CHAR[LT]),
        ),  # These must be escaped.
    )
    def test_spread_attr_escaped(self, html, escaped_char, expected):
        attrs = {"x": escaped_char}
        assert html(t"<div {attrs}></div>") == f'<div x="{expected}"></div>'

    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            (EQ, EQ),
            (BT, BT),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
           (SQ, ENT_BY_CHAR[SQ]),
           (DQ, ENT_BY_CHAR[DQ]),
           (AMP, ENT_BY_CHAR[AMP]),
           (GT, ENT_BY_CHAR[GT]),
           (LT, ENT_BY_CHAR[LT]),
        ),  # These must be escaped.
    )
    def test_data_attr_escaped(self, html, escaped_char, expected):
        data = {"x": escaped_char}
        assert html(t"<div data={data}></div>") == f'<div data-x="{expected}"></div>'

    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            (EQ, EQ),
            (BT, BT),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
           (SQ, ENT_BY_CHAR[SQ]),
           (DQ, ENT_BY_CHAR[DQ]),
           (AMP, ENT_BY_CHAR[AMP]),
           (GT, ENT_BY_CHAR[GT]),
           (LT, ENT_BY_CHAR[LT]),
        ),  # These must be escaped.
    )
    def test_aria_attr_escaped(self, html, escaped_char, expected):
        aria = {"x": escaped_char}
        assert html(t"<div aria={aria}></div>") == f'<div aria-x="{expected}"></div>'

    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            (EQ, EQ),
            (BT, BT),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
           (SQ, ENT_BY_CHAR[SQ]),
           (DQ, ENT_BY_CHAR[DQ]),
           (AMP, ENT_BY_CHAR[AMP]),
           (GT, ENT_BY_CHAR[GT]),
           (LT, ENT_BY_CHAR[LT]),
        ),  # These must be escaped.
    )
    def test_interpolated_escaped(self, html, escaped_char, expected):
        assert html(t"<div x={escaped_char}></div>") == f'<div x="{expected}"></div>'

    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            (EQ, EQ),
            (BT, BT),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
           (SQ, ENT_BY_CHAR[SQ]),
           (DQ, ENT_BY_CHAR[DQ]),
           (AMP, ENT_BY_CHAR[AMP]),
           (GT, ENT_BY_CHAR[GT]),
           (LT, ENT_BY_CHAR[LT]),
        ),  # These must be escaped.
    )
    def test_templated_escaped(self, html, escaped_char, expected):
        assert (
            html(t"<div x=prefix_{escaped_char}></div>")
            == f'<div x="prefix_{expected}"></div>'
        )


class TestLiteralRawTextPrematureClosing:
    """Tests for literal premature closing tag or partial closing part in text.

    This applies to all raw text: escapable and unescapable.

    This text can exist in tags: `TEXTAREA`, `TITLE`, `SCRIPT`, `STYLE`, etc.
    """

    @pytest.mark.parametrize(
        "trailing_char, tag",
        tuple(
            (c, t) for c in (TAB, LF, FF, CR, SOLIDUS) for t in ("title", "textarea", "script", "style")
        ),
    )
    @pytest.mark.xfail(reason="Known parsing issue: unparsed input discarded.")
    def test_cannot_contain_closing_tag(self, html, trailing_char, tag):
        """Partial versions of the closing tag are disallowed."""
        t = Template(f"<{tag}></{tag}{trailing_char}</{tag}>")
        with pytest.raises(ParsingError):
            _ = html(t)

    @pytest.mark.parametrize(
        "before, after, tag",
        tuple(
            (before, after, tag)
            for before in ("", "a")
            for after in ("", "a")
            for tag in ("title", "textarea", "script", "style")
        ),
    )
    def test_cannot_contain_closing_tag_gt(self, html, before, after, tag):
        """Test complete closing tag is not allowed."""
        with pytest.raises(ParsingError):
            _ = html(Template(f"<{tag}>{before}</{tag}>{after}</{tag}>"))


class TestDynamicRawTextPrematureClosing:
    """Tests for dynamic premature closing tag or partial closing part in text.

    This applies to all raw text: escapable and unescapable.

    This text can exist in tags: `TEXTAREA`, `TITLE`, `SCRIPT`, `STYLE`, etc.
    """

    @pytest.mark.parametrize(
        "trailing_char, tag",
        tuple(
            (c, t) for c in (TAB, LF, FF, CR, SOLIDUS, GT) for t in ("title", "textarea")
        ),
    )
    def test_escapable_mixed_cannot_contain_closing_tag(self, html, trailing_char, tag):
        """Partial versions of the closing tag are disallowed."""
        t = Template(f"<{tag}></{tag}") + t'{trailing_char}' + Template(f"</{tag}>")
        assert html(t) == f'<{tag}>{ENT_BY_CHAR[LT]}/{tag}{ENT_BY_CHAR.get(trailing_char, trailing_char)}</{tag}>'

    @pytest.mark.parametrize(
        "trailing_char",
        tuple(
            (c) for c in (TAB, LF, FF, CR, SOLIDUS, GT)
        ),
    )
    def test_script_mixed_cannot_contain_closing_tag(self, html, trailing_char):
        """Partial versions of the closing tag are disallowed."""
        t = t'<script></script{trailing_char}</script>'
        res = html(t)
        assert res == f"<script>" + HEX_LT + f"/script{trailing_char}</script>"
        assert res != f"<script></script{trailing_char}</script>"
