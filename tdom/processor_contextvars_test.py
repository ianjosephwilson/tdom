from contextvars import ContextVar
from dataclasses import dataclass
from string.templatelib import Template

from .processor import processor_service_factory
from .protocols import ComponentContextProvider

ThemeContext = ContextVar("ThemeContext", default="theme-default")


processor_api = processor_service_factory(slash_void=True, uppercase_doctype=True)


def html(*args, **kwargs):
    return processor_api.process_template(*args, **kwargs)


@dataclass(frozen=True)
class ThemeSetter:
    children: Template
    theme: str

    def get_context_values(self) -> tuple[tuple[ContextVar, object], ...]:
        return ((ThemeContext, self.theme),)

    def __call__(self) -> Template:
        return t"{self.children}"


def test_component_context_provider_protocol():
    assert isinstance(ThemeSetter, ComponentContextProvider)


def Header(msg: str) -> Template:
    return t'<div class="header"><{Logo} />{msg}</div>'


def Logo() -> Template:
    return t"<div class={ThemeContext.get()}>LOGO</div>"


def test_component_context_provider_set_value():
    theme = "theme-spring"

    assert html(
        t'<{ThemeSetter} theme="{theme}"><{Header} msg="Welcome To TDOM!" /></{ThemeSetter}>'
    ) == (
        '<div class="header"><div class="theme-spring">LOGO</div>Welcome To TDOM!</div>'
    )


def test_component_context_provider_default_value():
    assert html(t'<{Header} msg="Welcome To TDOM!" />') == (
        '<div class="header"><div class="theme-default">LOGO</div>Welcome To TDOM!</div>'
    )


def test_component_context_provider_clear_value():
    theme = "theme-spring"
    assert html(
        t'<{ThemeSetter} theme="{theme}"><{Header} msg="Welcome To TDOM!" /></{ThemeSetter}>'
        t'<{Header} msg="" />'
    ) == (
        '<div class="header"><div class="theme-spring">LOGO</div>Welcome To TDOM!</div>'
        '<div class="header"><div class="theme-default">LOGO</div></div>'
    )
