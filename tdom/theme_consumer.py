"""
Component users can get the theme provider and a utility function to get the theme supplied by an "extension".
"""

from string.templatelib import Template

from . import html
from .theme_provider import ThemeProvider, get_theme_cls


def Hello(theme_cls: str) -> Template:
    return t"<span class={theme_cls}>Hello</span>"


def HelloContainer() -> Template:
    return t"<div><{Hello} theme_cls={get_theme_cls:callback} /></div>"


def make_theme_example_t():
    return (
        t"<{ThemeProvider} theme='light'>"
        t"<{HelloContainer} />"  # Wrap in another component just to show separation.
        t"</{ThemeProvider}>"
    )


def test_theme_example():
    assert html(make_theme_example_t()) == ('<span class="light">Hello</span>')
