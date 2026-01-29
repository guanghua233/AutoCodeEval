import os
from typing import Dict, List, Optional, Tuple

_PRIMITIVES = {
    "byte", "short", "int", "long", "float", "double", "boolean", "char", "void"
}


def _strip_type(t: str) -> str:
    import re
    t = t.strip()
    t = re.sub(r"<.*?>", "", t)
    t = t.replace("...", "[]")
    t = t.replace("@", " ")
    t = " ".join(t.split())
    return t


def _simple_name(t: str) -> str:
    t = _strip_type(t)
    t = t.replace("[]", "")
    return t.split(".")[-1].strip()


class JavaProjectSearcherTS:
    def __init__(self):
        self.proj_dir: Optional[str] = None
        self.proj_info: Optional[Dict[str, Dict]] = None

        self.fqcn_to_file: Dict[str, str] = {}
        self.pkg_simple_to_fqcn: Dict[Tuple[str, str], str] = {}
        self.simple_to_fqcns: Dict[str, List[str]] = {}

        self.std_prefix = ("java.", "javax.", "jakarta.", "kotlin.", "scala.", "android.")
        self.std_packages = {"java.lang"}

    def set_proj(self, proj_dir: str, proj_info: Dict[str, Dict]):
        self.proj_dir = proj_dir if proj_dir.endswith(os.sep) else proj_dir + os.sep
        self.proj_info = proj_info
        self._build_indexes()

    def _build_indexes(self):
        self.fqcn_to_file.clear()
        self.pkg_simple_to_fqcn.clear()
        self.simple_to_fqcns.clear()

        for fpath, file_info in self.proj_info.items():
            mod = file_info.get("", {})
            pkg = mod.get("package") or ""

            for name, node in file_info.items():
                if name == "":
                    continue
                if "#" in name:
                    continue
                if "." in name:
                    continue
                if "$" in name:
                    continue
                if node.get("type") in ("Class", "Interface", "Enum", "Annotation", "Record"):
                    fqcn = f"{pkg}.{name}" if pkg else name
                    self.fqcn_to_file[fqcn] = fpath
                    self.pkg_simple_to_fqcn[(pkg, name)] = fqcn
                    self.simple_to_fqcns.setdefault(name, []).append(fqcn)

    def resolve_type(self, cur_file: str, type_name: str) -> Optional[str]:
        if not self.proj_info:
            return None

        t = _strip_type(type_name)
        if not t or t in _PRIMITIVES or t == "void":
            return None

        t_no_arr = t.replace("[]", "").strip()

        mod = self.proj_info[cur_file].get("", {})
        pkg = mod.get("package") or ""
        imports: List[str] = mod.get("imports", []) or []

        if "." in t_no_arr:
            if t_no_arr in self.fqcn_to_file:
                return t_no_arr

            cand = f"{pkg}.{t_no_arr}" if pkg else t_no_arr
            if cand in self.fqcn_to_file:
                return cand

            for imp in imports:
                if imp.endswith(".*"):
                    base = imp[:-2]
                    cand2 = f"{base}.{t_no_arr}"
                    if cand2 in self.fqcn_to_file:
                        return cand2

            return None

        simple = t_no_arr

        k = (pkg, simple)
        if k in self.pkg_simple_to_fqcn:
            return self.pkg_simple_to_fqcn[k]

        for imp in imports:
            if imp.endswith(".*"):
                continue
            if imp.endswith("." + simple) and imp in self.fqcn_to_file:
                return imp

        for imp in imports:
            if not imp.endswith(".*"):
                continue
            base = imp[:-2]
            cand = f"{base}.{simple}"
            if cand in self.fqcn_to_file:
                return cand

        cand = f"java.lang.{simple}"
        if cand in self.fqcn_to_file:
            return cand

        cands = self.simple_to_fqcns.get(simple, [])
        if len(cands) == 1:
            return cands[0]

        return None

    def type_to_file(self, fqcn: str) -> Optional[str]:
        return self.fqcn_to_file.get(fqcn)

    def resolve_import_to_file(self, cur_file: str, import_path: str) -> Optional[str]:
        if import_path.startswith(self.std_prefix):
            return None

        if import_path.endswith(".*"):
            base = import_path[:-2]
            for fqcn, fpath in self.fqcn_to_file.items():
                if fqcn.startswith(base + "."):
                    return fpath
            return None

        return self.fqcn_to_file.get(import_path)
