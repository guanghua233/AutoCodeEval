import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

from tree_sitter_languages import get_parser

_PRIMITIVES = {
    "byte", "short", "int", "long", "float", "double", "boolean", "char", "void"
}

_GENERIC_CLEAN = re.compile(r"<.*?>")
_ARR_CLEAN = re.compile(r"\[\]")


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _node_line(node) -> int:
    return int(node.start_point[0]) + 1


def _strip_type(type_str: str) -> str:
    t = type_str.strip()
    t = _GENERIC_CLEAN.sub("", t)
    t = t.replace("...", "[]")
    t = t.replace("@", " ")
    t = " ".join(t.split())
    return t


def _simple_name(type_str: str) -> str:
    t = _strip_type(type_str)
    t = _ARR_CLEAN.sub("", t).strip()
    return t.split(".")[-1].strip()


@dataclass
class JavaFileParseResult:
    module: Dict
    nodes: Dict[str, Dict]
    package: Optional[str]
    imports: List[str]
    top_types: List[str]
    inner_types: Dict[str, Set[str]]


class JavaTSParser:
    def __init__(self):
        self.parser = get_parser("java")

    def parse_file(self, fpath: str) -> JavaFileParseResult:
        with open(fpath, "rb") as f:
            src = f.read()

        tree = self.parser.parse(src)
        root = tree.root_node

        package = None
        imports: List[str] = []
        nodes: Dict[str, Dict] = {}

        module_node = {
            "type": "Module",
            "file_path": os.path.abspath(fpath),
        }

        top_types: List[str] = []
        inner_types: Dict[str, Set[str]] = {}

        for child in root.children:
            if child.type == "package_declaration":
                pkg = _text(src, child)
                pkg = pkg.replace("package", "").replace(";", "").strip()
                package = pkg
                module_node["package"] = package

            elif child.type == "import_declaration":
                imp = _text(src, child)
                imp = imp.strip().rstrip(";")
                imp = imp.replace("import", "", 1).strip()

                if imp.startswith("static "):
                    imp = imp[len("static "):].strip()
                    module_node.setdefault("static_imports", []).append(imp)
                    continue

                imports.append(imp)

            elif child.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "annotation_type_declaration",
                "record_declaration",
            ):
                self._parse_type_decl(src, child, nodes, top_types, inner_types, outer=None)

        module_node["imports"] = imports
        nodes[""] = module_node

        return JavaFileParseResult(
            module=module_node,
            nodes=nodes,
            package=package,
            imports=imports,
            top_types=top_types,
            inner_types=inner_types
        )

    def _parse_type_decl(
        self,
        src: bytes,
        node,
        nodes: Dict[str, Dict],
        top_types: List[str],
        inner_types: Dict[str, Set[str]],
        outer: Optional[str],
    ):
        name_node = None
        for c in node.children:
            if c.type == "identifier":
                name_node = c
                break
        if not name_node:
            return

        simple = _text(src, name_node)
        sline = _node_line(node)

        kind_map = {
            "class_declaration": "Class",
            "interface_declaration": "Interface",
            "enum_declaration": "Enum",
            "annotation_type_declaration": "Annotation",
            "record_declaration": "Record",
        }
        ntype = kind_map.get(node.type, "Class")

        qname = f"{outer}${simple}" if outer else simple

        if not outer:
            top_types.append(simple)
        else:
            inner_types.setdefault(outer, set()).add(simple)

        node_text = _text(src, node).strip()
        first_line = node_text.splitlines()[0] if node_text else ""

        entry = {
            "type": ntype,
            "def": first_line,
            "sline": sline,
        }

        rels = []
        for c in node.children:
            if c.type == "superclass":
                t = _strip_type(_text(src, c).replace("extends", "").strip())
                if t:
                    rels.append([t, "Extend"])
            elif c.type == "super_interfaces":
                txt = _text(src, c)
                txt = txt.replace("implements", "").strip()
                parts = [p.strip() for p in txt.split(",") if p.strip()]
                for p in parts:
                    rels.append([_strip_type(p), "Implement"])

        if rels:
            entry["rels"] = rels

        nodes[qname] = entry

        body = None
        for c in node.children:
            if c.type in ("class_body", "interface_body", "enum_body", "annotation_type_body"):
                body = c
                break
        if not body:
            return

        for m in body.children:
            if m.type == "field_declaration":
                self._parse_field(src, m, nodes, owner=qname)
            elif m.type == "method_declaration":
                self._parse_method(src, m, nodes, owner=qname)
            elif m.type == "constructor_declaration":
                self._parse_ctor(src, m, nodes, owner=qname)
            elif m.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "annotation_type_declaration",
                "record_declaration",
            ):
                self._parse_type_decl(src, m, nodes, top_types, inner_types, outer=qname)

    def _parse_field(self, src: bytes, node, nodes: Dict[str, Dict], owner: str):
        type_str = None
        declarators = []

        for c in node.children:
            if c.type in ("type", "integral_type", "floating_point_type", "boolean_type", "void_type"):
                type_str = _text(src, c)
            if c.type == "variable_declarator":
                declarators.append(c)

        if not declarators:
            return

        t = _strip_type(type_str) if type_str else None
        for d in declarators:
            name_node = None
            for cc in d.children:
                if cc.type == "identifier":
                    name_node = cc
                    break
            if not name_node:
                continue

            fname = _text(src, name_node)
            sline = _node_line(d)
            key = f"{owner}.{fname}"

            entry = {
                "type": "Field",
                "def": _text(src, node).splitlines()[0].strip(),
                "sline": sline,
                "in_class": owner,
            }

            if t and t not in _PRIMITIVES:
                entry["rels"] = [[t, "Typeof"]]

            nodes[key] = entry

    def _parse_method(self, src: bytes, node, nodes: Dict[str, Dict], owner: str):
        name = None
        ret_type = None

        for c in node.children:
            if c.type == "identifier" and name is None:
                name = _text(src, c)
            if c.type in ("type", "integral_type", "floating_point_type", "boolean_type", "void_type") and ret_type is None:
                ret_type = _text(src, c)

        if not name:
            return

        params = self._parse_params(src, node)
        sig = f"{owner}#{name}({', '.join(params)})"
        sline = _node_line(node)

        entry = {
            "type": "Method",
            "def": _text(src, node).splitlines()[0].strip(),
            "sline": sline,
            "in_class": owner,
        }

        rels = []
        rt = _strip_type(ret_type) if ret_type else None
        if rt and rt not in _PRIMITIVES and rt != "void":
            rels.append([rt, "Typeof"])

        if rels:
            entry["rels"] = rels

        nodes[sig] = entry

    def _parse_ctor(self, src: bytes, node, nodes: Dict[str, Dict], owner: str):
        name = None
        for c in node.children:
            if c.type == "identifier":
                name = _text(src, c)
                break

        params = self._parse_params(src, node)
        sig = f"{owner}#{name}({', '.join(params)})"
        sline = _node_line(node)

        entry = {
            "type": "Constructor",
            "def": _text(src, node).splitlines()[0].strip(),
            "sline": sline,
            "in_class": owner,
        }

        nodes[sig] = entry

    def _parse_params(self, src: bytes, node) -> List[str]:
        params: List[str] = []

        def find_formal_parameters(n):
            if n.type == "formal_parameters":
                return n
            for c in n.children:
                r = find_formal_parameters(c)
                if r:
                    return r
            return None

        fp = find_formal_parameters(node)
        if not fp:
            return params

        for c in fp.children:
            if c.type in ("formal_parameter", "receiver_parameter", "spread_parameter"):
                tnode = None
                for cc in c.children:
                    if cc.type in ("type", "integral_type", "floating_point_type", "boolean_type", "void_type"):
                        tnode = cc
                        break
                if tnode:
                    params.append(_strip_type(_text(src, tnode)))

        return params
