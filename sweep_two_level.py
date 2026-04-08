#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import subprocess
import sys
from typing import Iterable, List, Tuple


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def run(cmd: List[str], *, cwd: str = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def run_capture_stdout(cmd: List[str]) -> str:
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.stdout.strip()


def summarize_run_csv_row(row: str) -> Tuple[int, int]:
    """
    Simulator CSV fields (see harcom_superuser::~harcom_superuser):
      name,instructions,branches,condbr,npred,extra,div,div_at_end,misp,p1_lat,p2_lat,epi
    """
    parts = row.split(",")
    if len(parts) != 12:
        raise ValueError(f"Unexpected simulator CSV format ({len(parts)} fields): {row}")
    condbr = int(parts[3])
    misp = int(parts[8])
    return condbr, misp


def build_config_grid(pht_logs: Iterable[int], hist_lens: Iterable[int]) -> List[Tuple[int, int]]:
    configs: List[Tuple[int, int]] = []
    for h in hist_lens:
        for p in pht_logs:
            configs.append((p, h))
    configs = sorted(set(configs))
    return configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracedir", required=True, help="Directory containing '*_trace.gz' files.")
    ap.add_argument("--outdir", required=True, help="Output directory for sweep results.")
    ap.add_argument("--warmup", type=int, default=1_000_000)
    ap.add_argument("--siminst", type=int, default=40_000_000)
    ap.add_argument("--pht-logs", default="4,6,8", help="Comma-separated PHT_LOG values.")
    ap.add_argument("--hist-lens", default="2,4,6", help="Comma-separated HISTORY_LEN values.")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    compile_script = os.path.join(repo_root, "compile")
    cbp_bin = os.path.join(repo_root, "cbp")

    pht_logs = parse_int_list(args.pht_logs)
    hist_lens = parse_int_list(args.hist_lens)
    if not pht_logs or not hist_lens:
        print("Error: empty --pht-logs or --hist-lens", file=sys.stderr)
        return 2

    trace_files = sorted(glob.glob(os.path.join(args.tracedir, "*_trace.gz")))
    if not trace_files:
        print(f"Error: no '*_trace.gz' found in {args.tracedir}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    results_csv = os.path.join(args.outdir, "two_level_sweep_results.csv")

    configs = build_config_grid(pht_logs, hist_lens)
    print(f"Running {len(configs)} configurations on {len(trace_files)} traces.")

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pht_log", "history_len", "condbr_total", "misp_total", "accuracy"])

    for (pht_log, hist_len) in configs:
        pred = f"two_level<{pht_log},{hist_len}>"
        print(f"\n=== Config PHT_LOG={pht_log} HISTORY_LEN={hist_len} -> {pred} ===")

        # Compile this configuration
        run([compile_script, "cbp", f"-DPREDICTOR={pred}"], cwd=repo_root)

        cfg_outdir = os.path.join(args.outdir, f"pht{pht_log}_h{hist_len}")
        os.makedirs(cfg_outdir, exist_ok=True)

        condbr_total = 0
        misp_total = 0

        for trace_path in trace_files:
            trace_name = os.path.basename(trace_path)
            trace_name = trace_name[: -len("_trace.gz")]
            out = run_capture_stdout(
                [cbp_bin, trace_path, trace_name, str(args.warmup), str(args.siminst)]
            )

            out_file = os.path.join(cfg_outdir, f"{trace_name}.out")
            with open(out_file, "w") as tf:
                tf.write(out + "\n")

            row_condbr, row_misp = summarize_run_csv_row(out)
            condbr_total += row_condbr
            misp_total += row_misp

        accuracy = 1.0 - (misp_total / condbr_total if condbr_total else 0.0)
        print(f"Accuracy (level-2) = {accuracy:.6f}  (misp={misp_total} / condbr={condbr_total})")

        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([pht_log, hist_len, condbr_total, misp_total, f"{accuracy:.9f}"])

    print(f"\nDone. Wrote: {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())