"""
Security specific test suite.
"""

from string.templatelib import Template

import pytest

from .htmlspec import INVALID_ATTRIBUTE_NAME_CHARS, make_char_codepoint_reprs
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


NO_PARSER_TEST_ATTR_NAME_CHARS = (
    {"\u000c", "\r", "\n", "\t", " "}  # whitespace is collapsed
    | {"/"}  # these are removed in a lot of cases
    | {">"}  # this breaks parsing directly
)


class TestAttributeName:
    """
    Test for invalid characters in attribute names.

    Risk: Invalid characters could cause unexpected HTML rendering and allow for
    SCRIPT injection.

    Example: Most notably the '>' will cause a start tag to prematurely
    terminate within the attribute name, cause the remainder to be treated as
    regular html, ie. "evil><script src=example.com/evil.js>".

    Remediation: Raise an error when invalid characters are found in dynamic attribute names.

    Preferred: Always utilize a whitelist when working with dynamic attribute names.
    """

    @pytest.mark.parametrize("invalid_char", (chr(0x0000), "/", "=", ">"))
    def test_spread_specific_invalid_chars(self, html, invalid_char):
        """Test these specific chars in case our invalid set has a regression."""
        attrs = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div {attrs} />")

    @pytest.mark.parametrize("invalid_char", tuple(INVALID_ATTRIBUTE_NAME_CHARS))
    def test_spread_all_invalid_chars(self, html, invalid_char):
        attrs = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div {attrs} />")

    @pytest.mark.parametrize("invalid_char", (chr(0x0000), "/", "=", ">"))
    def test_aria_specific_invalid_chars(self, html, invalid_char):
        """Test these specific chars in case our invalid set has a regression."""
        aria = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            assert html(t"<div aria={aria} />")

    @pytest.mark.parametrize("invalid_char", tuple(INVALID_ATTRIBUTE_NAME_CHARS))
    def test_aria_all_invalid_chars(self, html, invalid_char):
        aria = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div aria={aria} />")

    @pytest.mark.parametrize("invalid_char", (chr(0x0000), "/", "=", ">"))
    def test_data_specific_invalid_chars(self, html, invalid_char):
        """Test these specific chars in case our invalid set has a regression."""
        data = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div data={data} />")

    @pytest.mark.parametrize("invalid_char", tuple(INVALID_ATTRIBUTE_NAME_CHARS))
    def test_data_all_invalid_chars(self, html, invalid_char):
        data = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div data={data} />")

    @pytest.mark.parametrize("invalid_char", (chr(0x0000), "="))
    def test_parser_specific_invalid_chars(self, html, invalid_char):
        """Test these specific chars in case our invalid set has a regression."""
        with pytest.raises(
            AttributeParsingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            print((invalid_char, html(Template(f"<div {invalid_char}='1' />"))))

    @pytest.mark.parametrize(
        "invalid_char",
        tuple(INVALID_ATTRIBUTE_NAME_CHARS - NO_PARSER_TEST_ATTR_NAME_CHARS),
    )
    def test_parser_all_invalid_chars(self, html, invalid_char):
        """Test these specific chars in case our invalid set has a regression."""
        with pytest.raises(
            AttributeParsingError,
            match=f"Attribute name contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            # With value
            print((invalid_char, html(Template(f"<div {invalid_char}='1' />"))))
        with pytest.raises(
            AttributeParsingError,
            match=f"Attribute name contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            # Boolean
            print((invalid_char, html(Template(f"<div {invalid_char} />"))))
