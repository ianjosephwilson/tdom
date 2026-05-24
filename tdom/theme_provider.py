"""
Component library or framework provides a set of tools to set the theme and get the theme.
"""

from contextvars import ContextVar
from string.templatelib import Template

from .context import ScopeProvider

ThemeCtx: ContextVar[str] = ContextVar("ThemeCtx", default="auto")


def get_theme_cls() -> str:
    """
    Get the theme when provided, otherwise return the default.
    """
    return ThemeCtx.get()


def ThemeProvider(children: Template, theme: str) -> ScopeProvider[str]:
    """
    Component which provides a theme to the given children.
    """
    return ScopeProvider(children=children, cvar=ThemeCtx, cvalue=theme)
