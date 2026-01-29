#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import signal
from argparse import ArgumentParser

from generator_java import JavaGenerator
from utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR, PT_FILE, MODEL


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timeout while processing")


def _norm_path(p: str) -> str:
    return p.replace("\\", "/")


def _resolve_abs_fpath(ds_repo_dir: str, pkg: str, fpath_in_ds: str) -> str:
    fp = _norm_path(fpath_in_ds).lstrip("/")
    if fp.startswith(_norm_path(pkg) + "/"):
        return os.path.join(ds_repo_dir, fp)
    return os.path.join(ds_repo_dir, pkg, fp)


def _append_jsonl(path: str, items: list):
    if not items:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            json.dump(it, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model", default=MODEL,
        help="Code model. Supported: deepseekcoder, codegen, codegen25, santacoder, starcoder, codellama, gpt35, gpt4"
    )
    parser.add_argument("-f", "--file", default=PT_FILE, help="Output prompt file path (jsonl, each line: {id,prompt})")
    parser.add_argument("--dataset", default=None, help="Dataset file path (jsonl). If not set, use DS_FILE")
    parser.add_argument("-t", "--timeout", type=int, default=30, help="Per-sample processing timeout (seconds)")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="Batch size: save results every N samples")
    parser.add_argument(
        "--noctx_file", default=None,
        help="Output file for samples with no retrieved context (has_ctx=False). Default: <file>.noctx"
    )
    args = parser.parse_args()

    print(f"Using model: {args.model}")
    print(f"Output prompt file: {args.file}")
    print(f"Dataset file: {args.dataset}")
    print(f"Per-sample timeout: {args.timeout} seconds")
    print(f"Batch size: {args.batch_size}")

    noctx_file = args.noctx_file if args.noctx_file else args.file + ".noctx"
    print(f"No-context sample output file: {noctx_file}")

    generator = JavaGenerator(DS_REPO_DIR, DS_GRAPH_DIR, args.model.lower())

    dataset_file = args.dataset if args.dataset else DS_FILE
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    print(f"Total {len(dataset)} samples to process")

    start_idx = 0
    if os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            processed = len([ln for ln in f if ln.strip()])
            if processed > 0:
                start_idx = processed
                print(f"Detected {args.file} already processed {processed} samples. Resuming from sample #{start_idx + 1}")

    ret = []
    noctx_samples = []
    timeout_samples = []

    signal.signal(signal.SIGALRM, timeout_handler)

    processed_cnt = 0
    skipped_cnt = 0
    failed_cnt = 0

    for i, item in enumerate(dataset[start_idx:], start=start_idx):
        if i % 10 == 0:
            print(f"Processing sample {i}/{len(dataset)}...")

        if i > start_idx and (i - start_idx) % args.batch_size == 0:
            print(f"Saving batch results... {i} samples completed")
            _append_jsonl(args.file, ret)
            _append_jsonl(noctx_file, noctx_samples)
            ret = []
            noctx_samples = []

        pkg = item["pkg"]
        abs_fpath = _resolve_abs_fpath(DS_REPO_DIR, pkg, item["fpath"])

        try:
            if abs_fpath.endswith(".java"):
                signal.alarm(args.timeout)

                start_time = time.time()

                out = generator.retrieve_prompt(pkg, abs_fpath, item["input"])
                try:
                    prompt_text, has_ctx = out
                except Exception:
                    prompt_text, has_ctx = out, (True if (out and str(out).strip()) else False)

                signal.alarm(0)

                process_time = time.time() - start_time
                if process_time > 5:
                    print(f"Sample {i} took a long time: {process_time:.2f} seconds")

                ret.append({
                    "id": item.get("id", i + 1),
                    "prompt": prompt_text
                })
                processed_cnt += 1

                if not has_ctx:
                    noctx_samples.append({
                        "id": item.get("id", i + 1),
                        "pkg": pkg,
                        "fpath": item.get("fpath", ""),
                        "abs_fpath": abs_fpath
                    })

            else:
                skipped_cnt += 1

        except TimeoutException:
            print(f"Warning: sample {i}, file {item['fpath']} timed out and was skipped")
            timeout_samples.append({"id": item.get("id", i + 1), "fpath": item["fpath"]})
            signal.alarm(0)
            continue

        except Exception as e:
            failed_cnt += 1
            print(f"Error while processing sample {i}, file {item['fpath']}: {repr(e)}")
            signal.alarm(0)
            continue

    _append_jsonl(args.file, ret)
    _append_jsonl(noctx_file, noctx_samples)

    if timeout_samples:
        timeout_file = args.file + ".timeout"
        os.makedirs(os.path.dirname(timeout_file) or ".", exist_ok=True)
        with open(timeout_file, "w", encoding="utf-8") as f:
            for it in timeout_samples:
                json.dump(it, f, ensure_ascii=False)
                f.write("\n")
        print(f"Timeout sample info saved to {timeout_file}")

    print("=== Done ===")
    print(f"Prompts generated successfully: {processed_cnt}")
    print(f"Non-Java samples skipped: {skipped_cnt}")
    print(f"Failed samples: {failed_cnt}")
    print(f"Timeout samples: {len(timeout_samples)}")
    print(
        f"No-context samples: {sum(1 for _ in open(noctx_file, 'r', encoding='utf-8')) if os.path.exists(noctx_file) else 0}"
    )
    print(f"Results saved to {args.file}")
    print(f"No-context samples saved to {noctx_file}")
