#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import signal
from argparse import ArgumentParser

from generator import CGenerator
from utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR, PT_FILE, MODEL


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timeout while processing")


def _append_jsonl(path: str, items: list):
    """Append a list of JSON-serializable dicts to a JSONL file."""
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
    parser.add_argument("-f", "--file", default=PT_FILE, help="Output prompt file path (jsonl)")
    parser.add_argument("-c", "--c_dataset", default=None, help="C dataset file path. If not set, use the default path.")
    parser.add_argument("-t", "--timeout", type=int, default=30, help="Per-sample processing timeout (seconds)")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="Batch size: save results every N samples")
    args = parser.parse_args()

    print(f"Using model: {args.model}")
    print(f"Output prompt file: {args.file}")
    print(f"C dataset file: {args.c_dataset}")
    print(f"Per-sample timeout: {args.timeout} seconds")
    print(f"Batch size: {args.batch_size}")

    generator = CGenerator(DS_REPO_DIR, DS_GRAPH_DIR, args.model.lower())

    dataset_file = args.c_dataset if args.c_dataset else DS_FILE
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

    if start_idx >= len(dataset):
        print(f"Detected {args.file} has already processed all {len(dataset)} samples. Nothing to do. Exiting.")
        exit(0)

    ret = []
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
            ret = []

        fpath = os.path.join(DS_REPO_DIR, item["fpath"])

        try:
            if fpath.endswith(".c") or fpath.endswith(".h"):
                signal.alarm(args.timeout)

                start_time = time.time()
                prompt_text = generator.retrieve_prompt(item["pkg"], fpath, item["input"])
                signal.alarm(0)

                process_time = time.time() - start_time
                if process_time > 5:
                    print(f"Sample {i} took a long time: {process_time:.2f} seconds")

                ret.append({
                    "id": item.get("id", i + 1),
                    "prompt": prompt_text
                })
                processed_cnt += 1

            else:
                skipped_cnt += 1
                print(f"Skipping non-C file: {fpath}")

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
    print(f"Non-C samples skipped: {skipped_cnt}")
    print(f"Failed samples: {failed_cnt}")
    print(f"Timeout samples: {len(timeout_samples)}")
    print(f"Results saved to {args.file}")
