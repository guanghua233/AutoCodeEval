import os
import json
import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from .tokenizer import CModelTokenizer
except:
    from tokenizer import CModelTokenizer


STD_LIB_PREFIXES = (
    "java.", "javax.", "jdk.", "sun.", "org.w3c.", "org.xml.", "jakarta."
)

IMPORT_LINE_RE = re.compile(
    r"^\s*import\s+(static\s+)?([a-zA-Z0-9_.]+(\.\*)?)\s*;\s*$", re.M
)


def _norm_path(p: str) -> str:
    return p.replace("\\", "/")


class JavaGenerator(object):
    def __init__(self, proj_dir: str, info_dir: str, model: str):
        self.proj_dir = os.path.abspath(proj_dir)
        self.info_dir = os.path.abspath(info_dir)
        self.tokenizer = CModelTokenizer(model)

        self.project: Optional[str] = None
        self.proj_info: Optional[Dict[str, Dict]] = None

    def _set_project(self, project: str):
        if project == self.project:
            return

        info_file = os.path.join(self.info_dir, f"{project}.json")
        if not os.path.isfile(info_file):
            print(f"Unknown project {project} in {self.info_dir}")
            self.project = project
            self.proj_info = None
            return

        self.project = project
        with open(info_file, "r", encoding="utf-8") as f:
            self.proj_info = json.load(f)

    def _project_root(self) -> str:
        return os.path.join(self.proj_dir, self.project) if self.project else self.proj_dir

    def _abs_to_rel_in_repo(self, abs_fpath: str) -> str:
        root = self._project_root()
        rel = os.path.relpath(abs_fpath, root)
        return _norm_path(rel)

    def get_suffix(self, fpath: str) -> str:
        fpath = fpath.replace("\\", "/")
        repo_root = self.proj_dir.replace("\\", "/").rstrip("/")

        if fpath.startswith(repo_root + "/"):
            fpath = fpath[len(repo_root) + 1:]

        return f"// path: {fpath}\n"

    def _extract_imports_from_source(self, source_code: str) -> List[str]:
        imports: List[str] = []
        for m in IMPORT_LINE_RE.finditer(source_code):
            fqn = m.group(2)
            if not fqn:
                continue
            if fqn.startswith(STD_LIB_PREFIXES):
                continue
            imports.append(fqn)
        return imports

    def _get_file_info(self, rel_path: str) -> Optional[Dict]:
        if not self.proj_info:
            return None
        return self.proj_info.get(rel_path)

    def _list_all_files_in_package(self, pkg: str) -> List[str]:
        if not self.proj_info:
            return []
        out = []
        for fpath, finfo in self.proj_info.items():
            mod = finfo.get("", {})
            if mod.get("package") == pkg:
                out.append(fpath)
        return sorted(set(out))

    def _resolve_imports_via_graph(self, cur_rel: str) -> List[str]:
        finfo = self._get_file_info(cur_rel)
        if not finfo:
            return []

        deps: Set[str] = set()

        for k, v in finfo.items():
            if not isinstance(k, str) or not k.startswith("import:"):
                continue
            inc = v.get("include")
            if isinstance(inc, list) and len(inc) >= 1 and inc[0]:
                deps.add(_norm_path(inc[0]))

        if deps:
            return sorted(deps)

        mod = finfo.get("", {})
        imports = mod.get("imports") or []
        for imp in imports:
            if not isinstance(imp, str):
                continue
            if imp.startswith(STD_LIB_PREFIXES):
                continue
            if imp.endswith(".*"):
                pkg = imp[:-2]
                for f in self._list_all_files_in_package(pkg):
                    deps.add(f)
            else:
                rel = imp.replace(".", "/") + ".java"
                for fpath in self.proj_info.keys():
                    fp = _norm_path(fpath)
                    if fp.endswith("/" + rel) or fp.endswith(rel):
                        deps.add(fp)
                        break

        return sorted(deps)

    def _collect_api_from_dep_file(self, dep_rel: str) -> str:
        finfo = self._get_file_info(dep_rel)
        if not finfo:
            return ""

        type_nodes: List[Tuple[int, str, Dict]] = []
        for k, v in finfo.items():
            if k == "" or not isinstance(k, str) or not isinstance(v, dict):
                continue
            if "." in k or "#" in k:
                continue
            if v.get("type") in ("Class", "Interface", "Enum", "Annotation", "Record"):
                sline = v.get("sline", 0) or 0
                type_nodes.append((sline, k, v))

        type_nodes.sort(key=lambda x: x[0])

        chunks: List[str] = []
        if type_nodes:
            chunks.append(f"// {dep_rel}")

        for _, tname, tinfo in type_nodes:
            tdef = (tinfo.get("def") or "").strip()
            if tdef:
                chunks.append(tdef)

            dot_prefix = tname + "."
            hash_prefix = tname + "#"

            mems: List[Tuple[int, str]] = []
            for k, v in finfo.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                if k.startswith(dot_prefix) and v.get("type") in ("Field",):
                    field_def = self._format_field_declaration(v.get("def") or "")
                    if field_def:
                        mems.append((v.get("sline", 0) or 0, field_def))
                elif k.startswith(hash_prefix) and v.get("type") in ("Method", "Constructor"):
                    method_def = self._format_method_declaration(v.get("def") or "")
                    if method_def:
                        mems.append((v.get("sline", 0) or 0, method_def))

            mems.sort(key=lambda x: x[0])
            for _, sig in mems:
                if sig:
                    chunks.append("    " + sig)

            chunks.append("")

        text = "\n".join(chunks).strip()
        return text + ("\n" if text else "")

    def _format_field_declaration(self, field_def: str) -> str:
        if not field_def:
            return ""
        s = field_def.strip()
        if s.startswith("@") and "(" not in s and " " not in s:
            return ""
        if "=" in s:
            s = s.split("=")[0].strip()
        if not s.endswith(";"):
            s += ";"
        return s

    def _format_method_declaration(self, method_def: str) -> str:
        if not method_def:
            return ""
        s = method_def.strip()
        if s.startswith("@") and s.endswith(";"):
            return ""
        if s.startswith("@") and "(" not in s and " " not in s:
            return ""
        if s.endswith("{"):
            s = s[:-1].strip() + ";"
        elif not s.endswith(";"):
            s += ";"
        return s

    def retrieve_prompt(self, project: str, abs_fpath: str, source_code: str):
        self._set_project(project)
        suffix = self.get_suffix(abs_fpath)

        if not self.proj_info:
            return self.tokenizer.truncate_concat(source_code, "", suffix), False

        cur_rel = self._abs_to_rel_in_repo(abs_fpath)
        dep_files = self._resolve_imports_via_graph(cur_rel)

        if not dep_files:
            imports = self._extract_imports_from_source(source_code)
            deps: Set[str] = set()
            for imp in imports:
                if imp.endswith(".*"):
                    for f in self._list_all_files_in_package(imp[:-2]):
                        deps.add(f)
                else:
                    rel = imp.replace(".", "/") + ".java"
                    for fpath in self.proj_info.keys():
                        fp = _norm_path(fpath)
                        if fp.endswith("/" + rel) or fp.endswith(rel):
                            deps.add(fp)
                            break
            dep_files = sorted(deps)

        prompt_parts: List[str] = []
        seen: Set[str] = set()
        for dep in dep_files:
            if dep in seen:
                continue
            seen.add(dep)
            api = self._collect_api_from_dep_file(dep)
            if api:
                prompt_parts.append(api)

        prompt = "\n".join(prompt_parts).strip()

        if not prompt:
            return self.tokenizer.truncate_concat(source_code, "", suffix), False

        max_prompt_length = self.tokenizer.cal_prompt_max_length(source_code, suffix)

        half_length = int(0.5 * self.tokenizer.max_input_length)
        if self.tokenizer.cal_token_nums(source_code) > half_length:
            source_code = source_code[-half_length:]

        if not self.tokenizer.judge_prompt(prompt, max_prompt_length):
            lines = prompt.split("\n")
            truncated_lines = []
            current_length = 0
            for line in lines:
                line_length = int(
                    self.tokenizer.tokenizer(line, return_tensors="pt").attention_mask.sum().item()
                )
                if current_length + line_length <= max_prompt_length:
                    truncated_lines.append(line)
                    current_length += line_length
                else:
                    break
            prompt = "\n".join(truncated_lines)

        return self.tokenizer.truncate_concat(source_code, prompt, suffix), True
