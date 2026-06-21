from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from string.templatelib import Interpolation, Template

from .htmlspec import VOID_ELEMENTS
from .placeholders import PlaceholderConfig, PlaceholderState
from .source import (
    EndTagSourceInfo,
    FrozenPosition,
    HTMLAttribute,
    Position,
    StartTagSourceInfo,
)
from .template_utils import TemplateRef, combine_template_refs
from .tnodes import (
    TAttribute,
    TComment,
    TComponent,
    TDocumentType,
    TElement,
    TFragment,
    TInterpolatedAttribute,
    TLiteralAttribute,
    TNode,
    TSpreadAttribute,
    TTemplatedAttribute,
    TText,
)


@dataclass()
class OpenTElement:
    tag: str
    attrs: tuple[TAttribute, ...]
    start_sinfo: StartTagSourceInfo
    children: list[TNode] = field(default_factory=list)


@dataclass()
class OpenTFragment:
    children: list[TNode] = field(default_factory=list)


@dataclass()
class OpenTComponent:
    start_i_index: int
    children_start_s_index: int
    """The strings index where the component's children template starts."""
    offset_into_children_start_s: int
    """The offset INTO the starting string where the component's children template starts."""
    attrs: tuple[TAttribute, ...]

    start_sinfo: StartTagSourceInfo

    # @NOTE: The `children` are discarded after parsing and are just used to
    # track template consistency or assist with error reporting.  If the
    # component is processed and returns its children template then that
    # template will be re-parsed (or pulled from the cache).
    children: list[TNode] = field(default_factory=list)


type OpenTag = OpenTElement | OpenTFragment | OpenTComponent


@dataclass
class SourceTracker:
    """Tracks template source and manages placeholders for the parser."""

    template: Template
    # if i_index >= s_index, feeding an interpolation;
    # otherwise, when i_index < s_index, feeding a string.
    i_index: int = -1  # The current interpolation index.
    s_index: int = -1  # The current string index.

    placeholders: PlaceholderState = field(default_factory=lambda: PlaceholderState())

    def __iter__(self):
        return self

    def __next__(self):
        if self.i_index < self.s_index:
            # Advance into the next interpolation UNLESS the last string
            # we returned was at the end of the template.
            if self.s_index == len(self.template.strings) - 1:
                raise StopIteration
            self.i_index += 1
            return self.placeholders.add_placeholder(self.i_index)
        elif self.i_index == self.s_index:
            # Advance into the next string
            self.s_index += 1
            return self.template.strings[self.s_index]
        else:
            raise AssertionError("{self.i_index=} should not exceed {self.s_index=}")

    def to_template_pos(self, parser_pos: FrozenPosition) -> FrozenPosition:
        """
        Translate the given parser (pos)ition into template (pos)ition.

        @NOTE: There can be newlines in an interpolation expression which
        results in the parser position's line being less than the
        template position's line since a placeholder will not contain newlines.

        @NOTE: Similarly an interpolation as displayed can be longer than a
        placeholder OR shorter than a placeholder causing the offsets to go
        out of sync.

        @NOTE: There is a weird issue with `format_spec` and the specification
        where you can't tell if a ':' was used or not when the `format_spec` is
        empty.  We just assume no one would leave it in without a non-empty
        format_spec, ie. t"{val:}" would not exist even though it is valid. The
        conversion does not have this issue because "{val!}" is invalid and when
        no conversion is set the conversion value is None.
        """
        #
        # Walk until we reach the given parser pos, keeping both parser position
        # and template position in sync.  When the given parser position is reached
        # then return the synced up template position.
        #
        pos = Position()
        tpos = Position()
        last_s_index = len(self.template.strings) - 1
        for s_index in range(len(self.template.strings)):
            #
            # Walk through `strings[s_index]`
            #
            s = self.template.strings[s_index]
            if parser_pos.line > pos.line:
                # need more lines
                nls_found = s.count("\n")  # how many were found?
                nls_need = parser_pos.line - pos.line  # how many are needed?
                if nls_found >= nls_need:
                    pos.line += nls_need
                    tpos.line += nls_need
                    offset_found = len(s.split("\n", nls_need + 1)[nls_need])
                    if offset_found >= parser_pos.offset:
                        # needed lines, found lines, found offset
                        tpos.offset = pos.offset = parser_pos.offset
                        return tpos.freeze()
                    else:
                        # got enough lines, still need more offset
                        tpos.offset = pos.offset = offset_found
                elif nls_found > 0:
                    # some lines but still need more lines
                    pos.line += nls_found
                    tpos.line += nls_found
                    tpos.offset = pos.offset = len(s[s.rfind("\n") + 1 :])
                else:
                    # no lines, still need more lines
                    offset_found = len(s)
                    tpos.offset += offset_found
                    pos.offset += offset_found
            elif parser_pos.line == pos.line:
                # got enough lines, we just need more offset
                offset_found = len(s[: s.find("\n")]) if "\n" in s else len(s)
                offset_need = parser_pos.offset - pos.offset
                if offset_found >= offset_need:
                    pos.offset += offset_need
                    tpos.offset += offset_need
                    # had lines, found offset
                    return tpos.freeze()
                else:
                    tpos.offset += offset_found
                    pos.offset += offset_found
            else:
                # We should have dropped out and failed earlier this would be a bug.
                raise AssertionError(
                    f"Unexpected line: {pos.line} greater than asked for {parser_pos.line}"
                )

            #
            # Walk through `interpolations[s_index]`
            #
            if s_index < last_s_index:
                ph_length = self.placeholders.measure_placeholder(s_index)
                if (
                    pos.line == parser_pos.line
                    and pos.offset + ph_length > parser_pos.offset
                ):
                    # Ie. we don't know how to determine how much of the
                    # interpolation expression would be equivalent to
                    # a substring of a placeholder.
                    raise ValueError(
                        f"Cannot split a placeholder for interpolations[{s_index}], placeholders are atomic."
                    )

                ip = self.template.interpolations[s_index]
                expr = ip.expression
                expr_line_count = expr.count("\n")
                tpos.line += expr_line_count
                pos.offset += ph_length
                EXCLAIMATION_POINT = CONVERSION_CHAR = SEMICOLON = LEFT_CURLY_BRACE = (
                    RIGHT_CURLY_BRACE
                ) = 1
                tail = (
                    (
                        EXCLAIMATION_POINT + CONVERSION_CHAR
                        if ip.conversion is not None
                        else 0
                    )  # "!" and conversion char or neither
                    + (SEMICOLON if ip.format_spec else 0)  # ":" or not
                    + len(ip.format_spec)
                    + RIGHT_CURLY_BRACE
                )
                if expr_line_count > 0:
                    tpos.offset = len(expr[expr.rfind("\n") + 1 :]) + tail
                else:
                    tpos.offset += LEFT_CURLY_BRACE + len(expr) + tail
                if pos == parser_pos:
                    return tpos.freeze()
        if pos == parser_pos:
            # @TODO: When can this fall through happen? Or is this always an error?
            return tpos.freeze()
        else:
            raise ValueError(
                "Unexpected position {pos}, did not reach required position {parser_pos}"
            )

    @property
    def interpolations(self) -> tuple[Interpolation, ...]:
        return self.template.interpolations

    def values_match(self, i_index1: int, i_index2: int) -> bool:
        return (
            self.interpolations[i_index1].value == self.interpolations[i_index2].value
        )

    def get_expression(
        self, i_index: int, fallback_prefix: str = "interpolation"
    ) -> str:
        """
        Resolve an interpolation index to its original expression for error messages.
        Falls back to a synthetic expression if the original is empty.
        """
        ip = self.interpolations[i_index]
        return ip.expression if ip.expression else f"{{{fallback_prefix}-{i_index}}}"

    def remake_fragment_str(self, ref: TemplateRef) -> str:
        """Remake a fragment of a template "as seen" in the template for error reporting."""
        return "".join(
            (part if isinstance(part, str) else self.remake_interpolation_str(part))
            for part in ref
        )

    def remake_interpolation_str(self, i_index: int) -> str:
        """Remake interpolation "as seen" in the template for error reporting."""
        ip = self.template.interpolations[i_index]
        expr_str = ip.expression
        conversion_str = f"!{ip.conversion}" if ip.conversion is not None else ""
        format_spec_str = f":{ip.format_spec}" if ip.format_spec else ""
        return f"{{{expr_str}{conversion_str}{format_spec_str}}}"

    def format_starttag(self, i_index: int) -> str:
        """Format a component start tag for error messages."""
        return self.get_expression(i_index, fallback_prefix="component-starttag")

    def format_endtag(self, i_index: int) -> str:
        return self.get_expression(i_index, fallback_prefix="component-endtag")


class TemplateParser(HTMLParser):
    root: OpenTFragment
    stack: list[OpenTag]
    source: SourceTracker | None

    closed_component_children: dict[TComponent, list[TNode]]
    "List of children for each closed component, stored at closing. "

    def __init__(self, *, convert_charrefs: bool = True):
        # This calls HTMLParser.reset() which we override to set up our state.
        super().__init__(convert_charrefs=convert_charrefs)

    # ------------------------------------------
    # Parse state helpers
    # ------------------------------------------

    def get_parent(self) -> OpenTag:
        """Return the current parent node to which new children should be added."""
        return self.stack[-1] if self.stack else self.root

    def append_child(self, child: TNode) -> None:
        parent = self.get_parent()
        parent.children.append(child)

    # ------------------------------------------
    # Attribute Helpers
    # ------------------------------------------

    def make_tattr(self, attr: HTMLAttribute) -> TAttribute:
        """Build a TAttribute from a raw attribute tuple."""
        source = self.get_source()
        name, value = attr

        name_ref = source.placeholders.remove_placeholders(name)
        value_ref = (
            source.placeholders.remove_placeholders(value)
            if value is not None
            else None
        )

        if name_ref.is_literal:
            if value_ref is None or value_ref.is_literal:
                return TLiteralAttribute(name=name, value=value)
            elif value_ref.is_singleton:
                return TInterpolatedAttribute(
                    name=name, value_i_index=value_ref.i_indexes[0]
                )
            else:
                return TTemplatedAttribute(name=name, value_ref=value_ref)
        if value_ref is not None:
            raise ValueError(
                "Attribute names cannot contain interpolations if the value is also interpolated."
            )
        if not name_ref.is_singleton:
            raise ValueError(
                "Spread attributes must have exactly one interpolation in the name."
            )
        return TSpreadAttribute(i_index=name_ref.i_indexes[0])

    def make_tattrs(self, attrs: Sequence[HTMLAttribute]) -> tuple[TAttribute, ...]:
        """Build TAttributes from raw attribute tuples."""
        return tuple(self.make_tattr(attr) for attr in attrs)

    # ------------------------------------------
    # Tag Helpers
    # ------------------------------------------

    def make_open_tag(
        self, tag: str, attrs: Sequence[HTMLAttribute], startend: bool = False
    ) -> OpenTag:
        """Build an OpenTag from a raw tag and attribute tuples."""
        source = self.get_source()
        tag_ref = source.placeholders.remove_placeholders(tag)
        if tag_ref.is_literal:
            open_tag = OpenTElement(
                tag=tag,
                attrs=self.make_tattrs(attrs),
                start_sinfo=StartTagSourceInfo(
                    starttag_text=self.get_starttag_text(),
                    raw_attrs=tuple(attrs),
                    startend=startend,
                    pos=self.get_parser_pos(),
                ),
            )
            return open_tag

        if not tag_ref.is_singleton:
            raise ValueError(
                "Component element tags must have exactly one interpolation."
            )

        # HERE BE DRAGONS: the interpolation at i_index should be a
        # component callable. We do not check this in the parser, instead
        # relying on higher layers to validate types and render correctly.
        i_index = tag_ref.i_indexes[0]

        # @NOTE: This must be stored when the tag is handled since it is
        # set based on when the template parts are fed in and otherwise
        # might be out of sync.
        # The starting s_index of the component's children template. Note that
        # this string either contains ">" or " />".  It might not be
        # i_index + 1 because attributes WITHIN the component's tag might
        # contain interpolations causing the i_index (and s_index) to advance
        # arbitrarily.
        children_start_s_index = self.get_source().s_index

        # @NOTE: This must be called when the tag is handled since it is
        # populated based on the most recently finished start tag. Otherwise
        # the value will be out of sync.
        starttag_text = self.get_starttag_text(
            f"Expected startag_text to be set when parsing component at {i_index}."
        )

        tattrs = self.make_tattrs(attrs)

        offset_into_children_start_s = self.compute_offset_into_children_start_s(
            start_i_index=i_index,
            tattrs=tattrs,
            config=source.placeholders.config,
            starttag_text=starttag_text,
        )

        open_tag = OpenTComponent(
            start_i_index=i_index,
            children_start_s_index=children_start_s_index,
            offset_into_children_start_s=offset_into_children_start_s,
            attrs=tattrs,
            start_sinfo=StartTagSourceInfo(
                starttag_text=self.get_starttag_text(),
                raw_attrs=tuple(attrs),
                startend=startend,
                pos=self.get_parser_pos(),
            ),
        )
        return open_tag

    def compute_offset_into_children_start_s(
        self,
        start_i_index: int,
        tattrs: tuple[TAttribute, ...],
        config: PlaceholderConfig,
        starttag_text: str,
    ) -> int:
        """
        Compute offset into "string" containing the start of children template.

        @NOTE: This is to actually OFFLOAD work to the parser itself.  If we try
        to "rebuild" the tag from the parse result we are bound to fail in some
        way(s). We essentially re-run the placeholder process but with content
        we KNOWN ends at the end of the starttag, ie. ">", because the parser
        told us that is where it ends (rather than trying to scan for ">"
        because ">" might be in literal tags).

        Examples:

        <{Comp}></{Comp}> -- len(">")
        <{Comp}>children</{Comp}> -- len(">")
        <{Comp} title="1>0">children</{Comp}> -- len(' title="1>0">')
        <{Comp} title="{'1>0'}">children</{Comp}> -- len('">')
        """
        # Rebuild known interpolations in the starttag.
        known: set[int] = {start_i_index}  # The component callable itself.
        for attr in tattrs:
            if isinstance(attr, TInterpolatedAttribute):
                known.add(attr.value_i_index)
            elif isinstance(attr, TSpreadAttribute):
                known.add(attr.i_index)
            elif isinstance(attr, TTemplatedAttribute):
                known.update(attr.value_ref.i_indexes)
        # Now re-remove those placeholders using the same config we used to
        # make them.
        temp_placeholders = PlaceholderState(known=known, config=config)
        tag_ref = temp_placeholders.remove_placeholders(starttag_text)
        if not temp_placeholders.is_empty:
            raise AssertionError(
                "There are extra placeholders still in the starttag_text."
            )
        # Now the last string should terminate the starttag and end with ">"
        # So this length is the offset from the last interpolation to the start
        # of the children's leading string.
        return len(tag_ref.strings[-1])

    def finalize_tag(
        self,
        open_tag: OpenTag,
        endtag_i_index: int | None = None,
        endtag_parser_pos: FrozenPosition | None = None,
    ) -> TNode:
        """Finalize an OpenTag into a TNode."""
        source = self.get_source()
        match open_tag:
            case OpenTElement(
                tag=tag, attrs=attrs, children=children, start_sinfo=start_sinfo
            ):
                tnode = TElement(
                    tag=tag,
                    attrs=attrs,
                    children=tuple(children),
                    start_sinfo=start_sinfo,
                    end_sinfo=EndTagSourceInfo(pos=endtag_parser_pos)
                    if endtag_parser_pos
                    else None,
                )
            case OpenTFragment(children=children):
                tnode = TFragment(children=tuple(children))
            case OpenTComponent(
                start_i_index=start_i_index,
                children_start_s_index=children_start_s_index,
                offset_into_children_start_s=offset_into_children_start_s,
                attrs=attrs,
                children=children,
                start_sinfo=start_sinfo,
            ):
                children_ref = self.extract_component_children_ref(
                    start_i_index=start_i_index,
                    endtag_i_index=endtag_i_index,
                    children_start_s_index=children_start_s_index,
                    offset_into_children_start_s=offset_into_children_start_s,
                    template=source.template,
                )
                tnode = TComponent(
                    start_i_index=start_i_index,
                    end_i_index=endtag_i_index,
                    children_ref=children_ref,
                    attrs=attrs,
                    start_sinfo=start_sinfo,
                    end_sinfo=EndTagSourceInfo(pos=endtag_parser_pos)
                    if endtag_parser_pos
                    else None,
                )
                self.closed_component_children[tnode] = (
                    children  # Save these for debugging.
                )
        return tnode

    def extract_component_children_ref(
        self,
        start_i_index: int,
        endtag_i_index: int | None,
        children_start_s_index: int,
        offset_into_children_start_s: int,
        template: Template,
    ) -> TemplateRef:
        """
        Extract the component children template from the entire template.

        We use this template as a "key" into the cache to get the TNode tree.
        """
        if start_i_index != endtag_i_index and endtag_i_index is not None:
            # CASE: <{Comp}>...</{Comp}> or <{Comp}></{Comp}>

            # Use the interpolation index of the callable in the closing tag
            # preceding "string" index is always the same as an interpolation index
            # The "string" should look like this: "...</"
            children_end_s_index = endtag_i_index
            # Offset past the trailing part of the component's start tag to get to
            # where the first "string" of the children's template starts.
            leading = template.strings[children_start_s_index][
                offset_into_children_start_s:
            ]
            if children_start_s_index == children_end_s_index:
                # CASE: Entire children template is a string, leading == trailing.
                leading = leading[: leading.rfind("</")]
                children_ref = TemplateRef(strings=(leading,), i_indexes=())
            else:
                # CASE: Children template contains interpolations so the trailing
                # "string" will not be the same as the leading "string".
                trailing = template.strings[children_end_s_index]
                trailing = trailing[: trailing.rfind("</")]
                children_ref = TemplateRef(
                    strings=(
                        leading,
                        *template.strings[
                            children_start_s_index + 1 : children_end_s_index
                        ],
                        trailing,
                    ),
                    i_indexes=tuple(
                        range(children_start_s_index, children_end_s_index)
                    ),
                )
        else:
            # CASE: <{Comp} /> -- no children template
            children_ref = TemplateRef(strings=("",), i_indexes=())
        return children_ref

    def validate_end_tag(self, tag: str, open_tag: OpenTag) -> int | None:
        """Validate that closing tag matches open tag. Return component end index if applicable."""
        source = self.get_source()
        tag_ref = source.placeholders.remove_placeholders(tag)

        match open_tag:
            case OpenTElement():
                if not tag_ref.is_literal:
                    raise ValueError(
                        f"Component closing tag found for element <{open_tag.tag}>."
                    )
                if tag != open_tag.tag:
                    raise ValueError(
                        f"Mismatched closing tag </{tag}> for element <{open_tag.tag}>."
                    )
                return None

            case OpenTFragment():
                raise NotImplementedError("We do not support anonymous fragments.")

            case OpenTComponent(start_i_index=start_i_index):
                if tag_ref.is_literal:
                    starttag = source.format_starttag(start_i_index)
                    e = ValueError(
                        f"Mismatched closing tag </{tag}> for component with tag {{{starttag}}}."
                    )
                    if self.has_ambiguous_forward_slash(open_tag.start_sinfo):
                        e.add_note(
                            f'Did you mean to quote the last attribute or put a space before "/>" for "<{{{starttag}}} .../>"?'
                        )
                    raise e
                if not tag_ref.is_singleton:
                    raise ValueError(
                        "Component end tags must have exactly one interpolation."
                    )
                return tag_ref.i_indexes[0]

    def get_starttag_text(self, msg: str = "Expecting starttag text to be set.") -> str:
        """
        Wrap get_starttag_text and just raise if None is returned.

        Do this so we don't guard for `None` everywhere.
        """
        starttag_text = super().get_starttag_text()
        if starttag_text is None:
            raise AssertionError(msg)
        return starttag_text

    def has_ambiguous_forward_slash(
        self, start_sinfo: StartTagSourceInfo | None
    ) -> bool:
        """
        Detect when an unquoted attribute value consumes a trailing "/" that
        *might* have been meant to attempt to self-close a tag, ie. "/>".

        This can come up with literal values or values with interpolations.

        Such as "<div title=test/>" or "<{Component} title=test/>".

        Or more often "<{Component} title={title}/>" which should be corrected
        with "<{Component} title={title} />".
        """
        if start_sinfo is not None:
            return (
                # has attributes
                len(start_sinfo.raw_attrs) > 0
                # last attr not bare attribute
                and start_sinfo.raw_attrs[-1][1] is not None
                # last char of last attr is "/"
                and start_sinfo.raw_attrs[-1][1][-1] == "/"
                # parsed starttag ends with "/>"
                and start_sinfo.starttag_text.endswith("/>")
                # if parsed as startend then its not ambiguous
                and not start_sinfo.startend
            )
        return False

    # ------------------------------------------
    # HTMLParser tag callbacks
    # ------------------------------------------

    def handle_starttag(self, tag: str, attrs: Sequence[HTMLAttribute]) -> None:
        open_tag = self.make_open_tag(tag, attrs)
        if isinstance(open_tag, OpenTElement) and open_tag.tag in VOID_ELEMENTS:
            final_tag = self.finalize_tag(open_tag)
            self.append_child(final_tag)
        else:
            self.stack.append(open_tag)

    def handle_startendtag(self, tag: str, attrs: Sequence[HTMLAttribute]) -> None:
        """Dispatch a self-closing tag, `<tag />` to specialized handlers."""
        open_tag = self.make_open_tag(tag, attrs, startend=True)
        final_tag = self.finalize_tag(open_tag)
        self.append_child(final_tag)

    def get_parser_pos(self) -> FrozenPosition:
        """
        Get the position of the parser.

        The content will be the t-string but with the interpolations replaced
        with placeholders.  Usually this is not very helpful and the position
        in the template (t-string) itself is preferred but this can be used
        to construct the template position.
        """
        line, offset = self.getpos()
        return FrozenPosition(line=line, offset=offset)

    def get_template_pos_msg(self) -> str:
        """
        Get the position in the template as if it read as a t-string.

        This can help find the locations of errors in the original t-string.
        """
        template_pos = self.get_source().to_template_pos(self.get_parser_pos())
        return f"line {template_pos.line} offset {template_pos.offset}"

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            source = self.get_source()
            tag_ref = source.placeholders.try_remove_placeholders(tag)
            if tag_ref.is_literal:
                raise ValueError(
                    f"Unexpected closing tag </{tag}> at {self.get_template_pos_msg()} with no open tag."
                )
            if not tag_ref.is_singleton:
                # @TODO: Also it doesn't match anything
                raise ValueError(
                    "Component end tags must have exactly one interpolation."
                )
            # Component tag endtag but no component tag is open...
            unmatched_endtag = source.format_endtag(tag_ref.i_indexes[0])
            raise ValueError(
                f"Unexpected closing component tag </{{{unmatched_endtag}}}> with no open tag."
            )
        open_tag = self.stack.pop()
        endtag_i_index = self.validate_end_tag(tag, open_tag)
        final_tag = self.finalize_tag(open_tag, endtag_i_index, self.get_parser_pos())
        self.append_child(final_tag)

    def get_closed_tcomps(
        self, root: OpenTag | None, recurse_component_children: bool = False
    ) -> list[TComponent]:
        """
        Get TComponents that were closed during parsing starting from `root`.

        If `root` is None then use the parser's default `root`.

        TComponents should be returned in the order they were closed in:
        from first closed to last closed.

        @NOTE: That the root is an `OpenTag` but its `children` are actually `TNode`s.
        """
        if root is None:
            root = self.root
        tcomps = []
        nodes = list(root.children)
        while nodes:
            node = nodes.pop()
            if isinstance(node, TComponent):
                tcomps.append(node)
                if recurse_component_children:
                    children = self.closed_component_children.get(node, [])
                    nodes.extend(children)
            elif isinstance(node, (TElement, TFragment)):
                nodes.extend(node.children)
        return tcomps

    # ------------------------------------------
    # HTMLParser other callbacks
    # ------------------------------------------

    def handle_data(self, data: str) -> None:
        source = self.get_source()
        ref = source.placeholders.remove_placeholders(data)
        parent = self.get_parent()
        if parent.children and isinstance(parent.children[-1], TText):
            parent.children[-1] = TText(
                ref=combine_template_refs(parent.children[-1].ref, ref)
            )
        else:
            self.append_child(TText(ref=ref))

    def handle_comment(self, data: str) -> None:
        source = self.get_source()
        ref = source.placeholders.remove_placeholders(data)
        comment = TComment(ref)
        self.append_child(comment)

    def handle_decl(self, decl: str) -> None:
        source = self.get_source()
        ref = source.placeholders.remove_placeholders(decl)
        if not ref.is_literal:
            raise ValueError("Interpolations are not allowed in declarations.")
        elif decl.upper().startswith("DOCTYPE "):
            doctype_content = decl[7:].strip()
            doctype = TDocumentType(doctype_content)
            self.append_child(doctype)
        else:
            raise NotImplementedError(
                "Only well formed DOCTYPE declarations are currently supported."
            )

    def reset(self):
        super().reset()
        self.root = OpenTFragment()
        self.stack = []
        self.source = None
        self.closed_component_children = {}

    def close(self) -> None:
        source = self.get_source()
        if self.waiting_for_data():
            # We apply heuristics here to try to guess why the parser didn't finish.
            if self.rawdata.count('"') % 2 == 1 or self.rawdata.count("'") % 2 == 1:
                raise ValueError(
                    "Parser expects more data, maybe you left an attribute quote unclosed?"
                )
            else:
                raise ValueError(
                    "Parser expects more data, is the template valid html?"
                )
        if self.stack:
            e = ValueError("Invalid HTML structure: unclosed tags remain.")
            # @TODO: We need to determine which tags this might apply to,
            # this only applies to components.
            parent = self.stack[-1]
            if isinstance(parent, OpenTComponent) and self.has_ambiguous_forward_slash(
                parent.start_sinfo
            ):
                # CASE: "<{C1} attr={value}/>" -- meant to self-close
                # Maybe user meant to self-close?
                starttag = source.format_starttag(parent.start_i_index)
                e.add_note(
                    f'Did you mean to quote the last attribute or put a space before "/>" for "<{{{starttag}}} .../>"?'
                )
            else:
                # CASE: t"<{C2}><{C1} attr=/></{C2}>"
                # Maybe user meant to self-close <{C1} ...>, but closed by </{C2}> leaving <{C2}...> open?
                # CASE: t"<{C3}><{C2}><{C1} attr=/></{C2}></{C3}>"
                for comp in reversed(
                    self.get_closed_tcomps(parent, recurse_component_children=True)
                ):
                    if (
                        comp.end_i_index is not None
                        and comp.start_i_index != comp.end_i_index
                        and not source.values_match(
                            comp.start_i_index, comp.end_i_index
                        )
                    ):
                        starttag = source.format_starttag(comp.start_i_index)
                        endtag = source.format_endtag(comp.end_i_index)
                        e.add_note(
                            f"Component start tag, <{{{starttag}}}>, and end tag, </{{{endtag}}}>, have values that do not match."
                        )
                        if self.has_ambiguous_forward_slash(comp.start_sinfo):
                            e.add_note(
                                f'Did you mean to quote the last attribute or put a space before "/>" for "<{{{starttag}}} .../>"?'
                            )
            raise e
        if not source.placeholders.is_empty:
            raise ValueError("Some placeholders were never resolved.")
        super().close()

    def waiting_for_data(self):
        return len(self.rawdata) > 0

    # ------------------------------------------
    # Getting the parsed node tree
    # ------------------------------------------

    def get_tnode(self) -> TNode:
        """Get the Node tree parsed from the input HTML."""
        # TODO: consider always returning a TTag?
        if len(self.root.children) > 1:
            # The parse structure results in multiple root elements, so we
            # return a Fragment to hold them all.
            return self.finalize_tag(self.root)
        elif len(self.root.children) == 1:
            # The parse structure results in a single root element, so we
            # return that element directly. This will be a non-Fragment Node.
            return self.root.children[0]
        else:
            # Special case: the parse structure is empty; we treat
            # this as an empty document fragment.
            # CONSIDER: or as an empty text node?
            return self.finalize_tag(self.root)

    # ------------------------------------------
    # Feeding and parsing
    # ------------------------------------------

    def get_source(self) -> SourceTracker:
        if self.source is None:
            raise AssertionError("Source has not been initialized.")
        return self.source

    def feed_template(self, template: Template) -> None:
        """Feed a Template's content to the parser."""
        assert self.source is None, "Did you forget to call reset?"
        self.source = SourceTracker(template)
        for content in self.source:
            self.feed(content)

    @staticmethod
    def parse(t: Template) -> TNode:
        """
        Parse a Template containing valid HTML and substitutions and return
        a TNode tree representing its structure. This cachable structure can later
        be resolved against actual interpolation values to produce a Node tree.
        """
        parser = TemplateParser()
        parser.feed_template(t)
        parser.close()
        return parser.get_tnode()
