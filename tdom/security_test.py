"""
Security specific test suite.
"""

from string.templatelib import Template

import pytest

from .htmlspec import INVALID_ATTRIBUTE_NAME_CHARS
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


class TestAttributeName:
    """
    Test escaping attribute names.
    """

    @pytest.mark.parametrize("invalid_char", (chr(0x0000), "/", "=", ">"))
    def test_specific_invalid_chars(self, html, invalid_char):
        """Test these specific chars in case our invalid set has a regression."""
        attrs = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {invalid_char}",
        ):
            assert html(t"<div {attrs} />")

    @pytest.mark.parametrize("invalid_char", tuple(INVALID_ATTRIBUTE_NAME_CHARS))
    def test_all_invalid_chars(self, html, invalid_char):
        attrs = {
            f"a{invalid_char}b": "1",
        }
        with pytest.raises(
            AttributeProcessingError,
            match=f"contains invalid characters: {invalid_char}",
        ):
            assert html(t"<div {attrs} />")
