import os
import time
import json

from utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR
from java_project_parser_ts import JavaProjectParserTS


def load_dataset_pkgs(ds_file: str) -> set:
    with open(ds_file, "r", encoding="utf-8", errors="ignore") as f:
        ds = [json.loads(line) for line in f if line.strip()]
    pkg_set = set(x["pkg"] for x in ds if "pkg" in x)
    return pkg_set


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        print(f"📁 Created output directory: {path}")


def main():
    t_all0 = time.perf_counter()

    pkg_set = load_dataset_pkgs(DS_FILE)
    print(f"📚 Dataset contains {len(pkg_set)} repositories")

    project_parser = JavaProjectParserTS()

    ensure_dir(DS_GRAPH_DIR)

    print(f"\n🚀 Start parsing repositories: {DS_REPO_DIR}\n")

    total = 0
    skipped = 0
    success = 0
    failed = 0

    for item in os.listdir(DS_REPO_DIR):
        if item not in pkg_set:
            continue

        dir_path = os.path.join(DS_REPO_DIR, item)
        if not os.path.isdir(dir_path):
            continue

        total += 1

        output_path = os.path.join(DS_GRAPH_DIR, f"{item}.json")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            skipped += 1
            print(f"⏭️ Skipping already processed repository: {item} (exists: {output_path})")
            continue

        print(f"\n==============================")
        print(f"🔷 Parsing repository: {item}")
        print(f"📂 Path: {dir_path}")
        print(f"==============================\n")

        t0 = time.perf_counter()
        try:
            info = project_parser.parse_dir(dir_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            dt = time.perf_counter() - t0
            success += 1
            print(f"💾 Graph file written: {output_path}")
            print(f"⏱️ Repository time: {dt:.2f}s")
        except Exception as e:
            dt = time.perf_counter() - t0
            failed += 1
            print(f"❌ Error while processing repository {item}: {e}")
            print(f"⏱️ Time before error: {dt:.2f}s")

    visible_files = [
        f for f in os.listdir(DS_GRAPH_DIR)
        if not f.startswith(".")
        and os.path.isfile(os.path.join(DS_GRAPH_DIR, f))
        and os.path.getsize(os.path.join(DS_GRAPH_DIR, f)) > 0
    ]

    dt_all = time.perf_counter() - t_all0

    print(f"\n🎉 Done! Generated {len(visible_files)} repository semantic graph files")
    print(f"📁 Output directory: {DS_GRAPH_DIR}")
    print(f"📄 File list: {visible_files}")

    print("\n📊 Statistics:")
    print(f"  - Total candidate repositories: {total}")
    print(f"  - Skipped (already processed): {skipped}")
    print(f"  - Parsed successfully: {success}")
    print(f"  - Failed: {failed}")
    print(f"⏱️ Total time: {dt_all:.2f}s")


if __name__ == "__main__":
    main()
