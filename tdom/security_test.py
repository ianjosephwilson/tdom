"""
Security specific test suite.

This doesn't exhaust every permuation possible but tries to make sure any
sort of internal validation checks have not been disabled.

### Approach

- Maintain independent constants/values from library code to prevent failures
in the library code from bypassing these checks.
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


DYNAMIC_INVALID_ATTR_NAME_CHARS = (chr(0x0000), "/", "=", ">", '"', "'")
"""
Notable subset of invalid attr name chars found during processing.
"""


LITERAL_INVALID_ATTR_NAME_CHARS = (chr(0x0000), "=", '"', "'")
"""
Notable subset of invalid attr name chars found during parsing.
"""


class TestAttributeName:
    """
    Test for invalid characters in attribute names.

    Risk: Invalid characters could cause unexpected HTML rendering and allow for
    SCRIPT injection.

    Example: Most notably the '>' will cause a start tag to prematurely
    terminate within the attribute name, cause the remainder to be treated as
    regular html, ie. "evil><script src=example.com/evil.js>".
    """

    @pytest.mark.parametrize("invalid_char", DYNAMIC_INVALID_ATTR_NAME_CHARS)
    def test_spread_attr(self, html, invalid_char):
        attrs = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div {attrs} />")

    @pytest.mark.parametrize("invalid_char", DYNAMIC_INVALID_ATTR_NAME_CHARS)
    def test_aria_attr(self, html, invalid_char):
        aria = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            assert html(t"<div aria={aria} />")

    @pytest.mark.parametrize("invalid_char", DYNAMIC_INVALID_ATTR_NAME_CHARS)
    def test_data_attr(self, html, invalid_char):
        data = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            _ = html(t"<div data={data} />")

    @pytest.mark.parametrize("invalid_char", LITERAL_INVALID_ATTR_NAME_CHARS)
    def test_literal_attr(self, html, invalid_char):
        """Test that invalid chars are caught using a few specific examples."""
        with pytest.raises(
            AttributeParsingError,
            match=f"contains invalid characters: {make_char_codepoint_reprs(invalid_char)}",
        ):
            # We use `Template` here because we are testing the parsing of static text.
            _ = html(Template(f"<div {invalid_char}=1 />"))
