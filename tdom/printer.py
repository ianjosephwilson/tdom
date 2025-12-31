from dataclasses import dataclass

from markupsafe import escape as escape_html_text

from .nodes import (
    VOID_ELEMENTS,
)
from .tnodes import (
    TNode,
    TElement,
    TComponent,
    TFragment,
    TText,
    TComment,
    TDocumentType,
)
from .escaping import escape_html_script, escape_html_style, escape_html_comment


def get_parent_tag_for_children(
    node: TComponent | TElement | TFragment, parent_tag: str | None
) -> str | None:
    match node:
        case TElement():
            return node.tag
        case TFragment():
            return parent_tag
        case TComponent():
            return parent_tag
        case _:
            raise TypeError(
                "Cannot resolve parent tag for children of tnodes other than TElement and TFragment."
            )


def get_starttag(node: TComponent | TElement | TFragment) -> str:
    match node:
        case TElement():
            attrs_str = f" {repr(node.attrs)}" if node.attrs else ""
            return f"<{node.tag}{attrs_str}>"
        case TComponent():
            return f"<component-at-index-{node.start_i_index} {repr(node.attrs)}>"
        case TFragment():
            return "<tnode-fragment>"
        case _:
            raise TypeError(
                "Cannot resolve start tag for tnodes other than TElement and TFragment."
            )


def get_endtag(node: TComponent | TElement | TFragment) -> str:
    match node:
        case TElement():
            return f"</{node.tag}>"
        case TComponent():
            return f"</component-at-index-{node.start_i_index}>"
        case TFragment():
            return "</tnode-fragment>"
        case _:
            raise TypeError(
                "Cannot resolve end tag for tnodes other than TElement and TFragment."
            )


@dataclass
class Symbol:
    symbol_str: str


def get_nl():
    """Get a newline character."""
    # @TODO: This is A way but not THE way...
    return "\n"


def pformat_tnode(
    tnode: TNode,
    start_level: int = 0,
    start_parent_tag: str | None = None,
    show_text=False,
    show_ws_only=False,
    text_marker="TEXT",
    ws_marker="WS_ONLY",
) -> list[str]:
    """
    Try to pretty format a tnode into a list of str chunks.

    tnode:
        The node to start printing from.
    start_level:
        The indent level to start at.
    start_parent_tag:
        Starting parent tag to enforce escaping, defaults to markupsafe.
    show_text:
        Show actual text in the text nodes, otherwse show a marker.
    show_ws_only:
        Show actual text or markers for text nodes that are whitespace
        only otherwise omit them.
    text_marker:
        Marker to use when not showing actual text and there is non-whitespace content.
    ws_marker:
        Marker to use when not showing actual text and there is only whitespace.
    """
    q: list[tuple[int, str | None, TNode | Symbol]] = [
        (start_level, start_parent_tag, tnode)
    ]
    out: list[str] = []

    def append_out(text: str, level: int, with_nl=True) -> None:
        out.append("{}{}{}".format("  " * level, text, get_nl() if with_nl else ""))

    while q:
        level, parent_tag, node = q.pop()
        match node:
            case TDocumentType(text):
                append_out(f"<!DOCTYPE {text}>", level)
            case Symbol():
                append_out(node.symbol_str, level)
            case TComponent():
                append_out(get_starttag(node), level)
                if node.start_i_index != node.end_i_index and node.end_i_index is not None:
                    q.append((level, None, Symbol(get_endtag(node))))
                    q.extend(
                        [
                            (
                                level + 1,
                                get_parent_tag_for_children(node, parent_tag),
                                child,
                            )
                            for child in reversed(node.children)
                        ]
                    )
            case TElement():
                append_out(get_starttag(node), level)
                if node.tag not in VOID_ELEMENTS:
                    # Maybe try something here when text is the last child or something
                    # q.append((0, None, Symbol(get_nl())))
                    q.append((level, None, Symbol(get_endtag(node))))
                    q.extend(
                        [
                            (
                                level + 1,
                                get_parent_tag_for_children(node, parent_tag),
                                child,
                            )
                            for child in reversed(node.children)
                        ]
                    )
            case TFragment():
                append_out(get_starttag(node), level)
                if node.children:
                    q.append((level, None, Symbol(get_endtag(node))))
                    q.extend(
                        [
                            (
                                level + 1,
                                get_parent_tag_for_children(node, parent_tag),
                                child,
                            )
                            for child in reversed(node.children)
                        ]
                    )
            case TText() | TComment():
                if isinstance(node, TText):
                    if parent_tag == "script":
                        formatter = escape_html_script
                    elif parent_tag == "style":
                        formatter = escape_html_style
                    else:
                        formatter = escape_html_text
                else:
                    formatter = escape_html_comment
                count = len(list(node.ref))
                for index, part in enumerate(node.ref):
                    indent = level if index == 0 else 0
                    if isinstance(part, str):
                        if show_text:
                            if show_ws_only or part.strip():
                                text = formatter(part)
                            else:
                                text = None
                        else:
                            if part.strip():
                                text = text_marker
                            else:
                                if show_ws_only:
                                    text = ws_marker
                                else:
                                    text = None
                        text_fmt = "<text>{}</text>"
                    else:
                        text_fmt = "<slot>{}</slot>"
                        text = "{" + str(part) + "}"
                    if text is not None:
                        append_out(text_fmt.format(text), indent, with_nl=False)
                        if index == count - 1:
                            append_out("", 0)
    return out
