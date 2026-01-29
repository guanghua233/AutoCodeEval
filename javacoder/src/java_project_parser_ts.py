import os
import re
from typing import Dict, Set, List

from javafile_parse_ts import JavaTSParser
from node_prompt_java_ts import JavaProjectSearcherTS


class JavaProjectParserTS:
    def __init__(self):
        self.parser = JavaTSParser()
        self.searcher = JavaProjectSearcherTS()

        self.file_pattern = re.compile(r"[^\w\-]")
        self.java_pattern = re.compile(r"\.java$")

        self.proj_dir = None
        self.parse_res: Dict[str, Dict] = {}

        self.EXPAND_WILDCARD_IMPORTS = False
        self.WILDCARD_IMPORT_MAX_MEMBERS = 30

    def set_proj_dir(self, dir_path: str):
        self.proj_dir = dir_path if dir_path.endswith(os.sep) else dir_path + os.sep
        print(f"\n📁 Set project directory: {self.proj_dir}")

    def _get_all_java_paths(self, target_path: str) -> Set[str]:
        print(f"🔍 Scanning Java files: {target_path}")
        if not os.path.isdir(target_path):
            print("❌ Directory does not exist!")
            return set()

        stack = [target_path]
        java_files: Set[str] = set()

        while stack:
            d = stack.pop()
            for item in os.listdir(d):
                if item.startswith("."):
                    continue
                fpath = os.path.join(d, item)

                if os.path.isdir(fpath):
                    if re.search(self.file_pattern, item) is None:
                        stack.append(fpath)
                else:
                    if os.path.isfile(fpath) and self.java_pattern.search(fpath):
                        base = os.path.splitext(item)[0]
                        if re.search(self.file_pattern, base) is None:
                            java_files.add(fpath)

        return java_files

    def _module_name(self, fpath: str) -> str:
        return fpath[len(self.proj_dir):]

    def parse_dir(self, proj_dir: str) -> Dict[str, Dict]:
        self.set_proj_dir(proj_dir)
        java_files = sorted(self._get_all_java_paths(proj_dir))
        print(f"📄 Found {len(java_files)} Java source files")

        self.parse_res = {}

        for fpath in java_files:
            module = self._module_name(fpath)
            try:
                res = self.parser.parse_file(fpath).nodes
                if res:
                    self.parse_res[module] = res
            except Exception as e:
                print(f"     ❌ Parse failed: {module}: {e}")

        self.searcher.set_proj(proj_dir, self.parse_res)
        self._resolve_project_types()

        print(f"✅ Directory parsing completed: {proj_dir}")
        print(f"📦 Parsed {len(self.parse_res)} file nodes in total\n")
        return self.parse_res

    def _resolve_project_types(self):
        print("🔧 Resolving cross-file types and cleaning relations...")

        for module, file_info in self.parse_res.items():
            mod = file_info.get("", {})
            imports = mod.get("imports", []) or []

            self._inject_import_virtual_nodes(module, file_info, imports)

            for name, node in list(file_info.items()):
                rels = node.get("rels")
                if not rels:
                    continue

                new_rels: List[List[str]] = []
                for r in rels:
                    if not r or len(r) < 2:
                        continue
                    tgt, rtype = r[0], r[1]

                    if rtype in ("Typeof", "TypeofParam", "TypeofLocal", "Extend", "Implement", "New"):
                        fqcn = self.searcher.resolve_type(module, tgt)
                        if fqcn:
                            new_rels.append([fqcn, rtype])
                        else:
                            new_rels.append([tgt, rtype])
                        continue

                    if rtype == "Call":
                        rewritten = self.searcher.resolve_call(module, tgt)
                        if rewritten:
                            new_rels.append([rewritten, rtype])
                        else:
                            new_rels.append([tgt, rtype])
                        continue

                    if rtype == "Access":
                        rewritten = self.searcher.resolve_access(module, tgt)
                        if rewritten:
                            new_rels.append([rewritten, rtype])
                        else:
                            new_rels.append([tgt, rtype])
                        continue

                    new_rels.append([tgt, rtype])

                node["rels"] = self._dedup_rels(new_rels)

    def _inject_import_virtual_nodes(self, module: str, file_info: Dict, imports: List[str]):
        for imp in imports:
            target = self.searcher.resolve_import_to_file(module, imp)
            if target:
                file_info[f"import:{imp}"] = {
                    "type": "Variable",
                    "def": f"import {imp};",
                    "sline": -1,
                    "include": [target, None],
                }

            if self.EXPAND_WILDCARD_IMPORTS and imp.endswith(".*"):
                pkg_prefix = imp[:-2]
                members = self.searcher.package_members(pkg_prefix)

                if not members:
                    continue

                members = members[: self.WILDCARD_IMPORT_MAX_MEMBERS]
                for fqcn, fpath in members:
                    key = f"import:{imp}:{fqcn}"
                    file_info[key] = {
                        "type": "Variable",
                        "def": f"import {imp}; -> {fqcn}",
                        "sline": -1,
                        "include": [fpath, None],
                    }

    def _dedup_rels(self, rels: List[List[str]]) -> List[List[str]]:
        seen = set()
        out: List[List[str]] = []
        for r in rels:
            if not r or len(r) < 2:
                continue
            tgt, rtype = r[0], r[1]
            key = (tgt, rtype)
            if key in seen:
                continue
            seen.add(key)
            out.append([tgt, rtype])
        return out
