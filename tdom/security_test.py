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

import pytest

from .htmlspec import make_char_codepoint_reprs
from .parser import AttributeParsingError
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


class TestDynamicAttributeName:
    @pytest.mark.parametrize("invalid_char", (chr(0x0000), "/", "=", ">", '"', "'"))
    def test_spread_attr(self, html, invalid_char):
        attrs = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div {attrs} />")

    @pytest.mark.parametrize("invalid_char", (chr(0x0000), "/", "=", ">", '"', "'"))
    def test_aria_attr(self, html, invalid_char):
        aria = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            assert html(t"<div aria={aria} />")

    @pytest.mark.parametrize("invalid_char", (chr(0x0000), "/", "=", ">", '"', "'"))
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
    @pytest.mark.parametrize("invalid_char", (chr(0x0000), "=", '"', "'"))
    def test_literal_attr(self, html, invalid_char):
        """Test that invalid chars are caught using a few specific examples."""
        with pytest.raises(
            AttributeParsingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            # We use `Template` here because we are testing the parsing of static text.
            _ = html(Template(f"<div {invalid_char}=1 />"))


class TestLiteralAttributeValue:
    """
    Certain characters must be wrapped in quotes and/or escaped.
    """

    def test_unquoted_double_quote(self, html):
        assert (
            html(t'<div x="></div>') == ""
        )  # This is currently not working, but not a security issue.

    def test_quoted_dangling_double_quote(self, html):
        assert html(t"""<div x='"'></div>""") == '<div x="&#34;"></div>'

    @pytest.mark.skip(
        "HTMLParser chokes on dangling quotes and keeps waiting for more data."
    )
    def test_dangling_double_quote(self, html):
        assert html(t'<div x="></div>') == """<div x='"'></div>"""  # or escaped

    def test_unquoted_single_quote(self, html):
        assert (
            html(t"<div x='></div>") == ""
        )  # This is currently not working, but not a security issue.

    def test_quoted_single_quote(self, html):
        assert html(t"""<div x="'"></div>""") == '<div x="&#39;"></div>'

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
            == '<div x="&amp;"></div>'
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
            == '<div x="&lt;"></div>'
        )

    @pytest.mark.parametrize(
        "quoting", ("'", '"')
    )  # @NOTE: We cannot test > without quotes because the parsing does not work.
    def test_greaterthan(self, html, quoting):
        assert (
            html(Template(f"<div x={quoting}>{quoting}></div>"))
            == '<div x="&gt;"></div>'
        )

    def test_unquoted_greaterthan(self, html):
        """The parser should interpret the first > as the end of the tag so the following > is treated as text content and escaped."""
        assert html(Template("<div x=>></div>")) == '<div x="">&gt;</div>'


class TestDynamicAttributeValue:
    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            ("=", "="),
            ("`", "`"),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
            ("'", "&#39;"),
            ('"', "&#34;"),
            ("&", "&amp;"),
            (">", "&gt;"),
            ("<", "&lt;"),
        ),  # These must be escaped.
    )
    def test_spread_attr_escaped(self, html, escaped_char, expected):
        attrs = {"x": escaped_char}
        assert html(t"<div {attrs}></div>") == f'<div x="{expected}"></div>'

    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            ("=", "="),
            ("`", "`"),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
            ("'", "&#39;"),
            ('"', "&#34;"),
            ("&", "&amp;"),
            (">", "&gt;"),
            ("<", "&lt;"),
        ),  # These must be escaped.
    )
    def test_data_attr_escaped(self, html, escaped_char, expected):
        data = {"x": escaped_char}
        assert html(t"<div data={data}></div>") == f'<div data-x="{expected}"></div>'

    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            ("=", "="),
            ("`", "`"),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
            ("'", "&#39;"),
            ('"', "&#34;"),
            ("&", "&amp;"),
            (">", "&gt;"),
            ("<", "&lt;"),
        ),  # These must be escaped.
    )
    def test_aria_attr_escaped(self, html, escaped_char, expected):
        aria = {"x": escaped_char}
        assert html(t"<div aria={aria}></div>") == f'<div aria-x="{expected}"></div>'

    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            ("=", "="),
            ("`", "`"),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
            ("'", "&#39;"),
            ('"', "&#34;"),
            ("&", "&amp;"),
            (">", "&gt;"),
            ("<", "&lt;"),
        ),  # These must be escaped.
    )
    def test_interpolated_escaped(self, html, escaped_char, expected):
        assert html(t"<div x={escaped_char}></div>") == f'<div x="{expected}"></div>'

    @pytest.mark.parametrize(
        "escaped_char, expected",
        (
            ("=", "="),
            ("`", "`"),
        )  # These are not escaped, but must be wrapped in quotes.
        + (
            ("'", "&#39;"),
            ('"', "&#34;"),
            ("&", "&amp;"),
            (">", "&gt;"),
            ("<", "&lt;"),
        ),  # These must be escaped.
    )
    def test_templated_escaped(self, html, escaped_char, expected):
        assert (
            html(t"<div x=prefix_{escaped_char}></div>")
            == f'<div x="prefix_{expected}"></div>'
        )
