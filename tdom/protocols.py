import typing as t
from contextvars import ContextVar
from string.templatelib import Template


@t.runtime_checkable
class HasHTMLDunder(t.Protocol):
    def __html__(self) -> str: ...  # pragma: no cover


@t.runtime_checkable
class ComponentObject(t.Protocol):
    def __call__(self) -> Template: ...


@t.runtime_checkable
class ComponentContextProvider(ComponentObject, t.Protocol):
    def get_context_values(self) -> tuple[tuple[ContextVar, object], ...]: ...
