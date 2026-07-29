"""Score every Big-Math problem with OSIM to produce an `osim_solve_rate`
column, so the dataset can be re-filtered for OUR probe model.

Why: rd211/Big-Math-RL-Verified-Filtered is already difficulty-filtered, but
against Llama-3.1-8B -- its surviving `llama8b_solve_rate` band is
0.0156 (=1/64) to 0.5938 (=38/64), i.e. consistent with `0 < rate < 0.6`.
Both ends of that rule exist to avoid zero gradient: a problem the probe never
solves cannot show a learning delta (floor), and one it usually solves has no
headroom (ceiling).

That band is a property of the PROBE MODEL, not of the problems. OSIM is
Qwen3-8B-based and much stronger at math than Llama-3.1-8B (measured: 0.87 and
0.91 correct on the two upper llama bins), so much of this set is at ceiling
for OSIM despite being correctly filtered for Llama. Re-applying the same rule
with OSIM's own rate restores the intended selection.

Sampling matches the eval knowledge probe exactly (temperature 0.6,
number_student_attempts=8, max_tokens_per_student_attempt=3900, and the same
neutral `student_initial_attempt_prompt.txt`) so `osim_solve_rate` is directly
comparable to `llama8b_solve_rate` rather than to a differently-decoded number.

Deliberately OUT of persona: the passive persona declines the probe 15.6% of
the time ("I don't know where to start... can you just show me?"), and those
refusals score as wrong, which would conflate "cannot solve" with "would not
attempt". Persona applies to in-dialogue turns only.

Usage:
    CUDA_VISIBLE_DEVICES=4 python utils/osim_score_dataset.py \
        --out logs/osim_solve_rates.json
"""

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from jinja2 import Template
from vllm import LLM, SamplingParams

from src.utils.utils import check_equal, extract_answer

OSIM_MODEL = "cmu-lti/osim-8b"
DATASET = "rd211/Big-Math-RL-Verified-Filtered"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The same neutral probe prompt the Llama knowledge probe uses, so the rate is
# measured under identical instructions.
PROMPT_PATH = os.path.join(ROOT, "prompt_templates",
                           "student_initial_attempt_prompt.txt")

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)

# TutorRL's inferred selection rule, re-applied to whichever probe model's rate
# is being used. Inferred from the surviving band of llama8b_solve_rate (the
# dataset card documents no criterion and the source dataset is gated), but the
# bounds land exactly on 1/64 and 38/64, which pins it tightly.
KEEP_LO, KEEP_HI = 0.0, 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=-1,
                    help="score only the first N problems (-1 = all)")
    ap.add_argument("--samples", type=int, default=8,
                    help="matches generation.number_student_attempts")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="matches the eval student_model vllm temperature")
    ap.add_argument("--max-tokens", type=int, default=3900,
                    help="matches generation.max_tokens_per_student_attempt")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", default="logs/osim_solve_rates.json")
    args = ap.parse_args()

    with open(PROMPT_PATH) as f:
        tpl = Template(f.read())

    ds = load_dataset(DATASET, split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Scoring {len(ds)} problems x {args.samples} samples "
          f"= {len(ds) * args.samples} generations", flush=True)

    prompts = [
        [{"role": "user", "content": tpl.render(problem=p)}]
        for p in ds["problem"]
    ]

    llm = LLM(
        model=OSIM_MODEL,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    # n=args.samples shares one prefill across all attempts for a problem --
    # materially cheaper than duplicating the prompt `samples` times.
    sp = SamplingParams(n=args.samples, temperature=args.temperature,
                        top_p=1.0, max_tokens=args.max_tokens)

    try:
        outs = llm.chat(prompts, sp, chat_template_kwargs={"enable_thinking": False})
    except Exception as e:
        print(f"enable_thinking kwarg rejected ({e}); retrying without it")
        outs = llm.chat(prompts, sp)

    records = []
    for i, o in enumerate(outs):
        row = ds[i]
        n_correct = n_boxed = n_trunc = 0
        for comp in o.outputs:
            text = _THINK_RE.sub("", comp.text).strip()
            boxed = extract_answer(text)
            n_trunc += comp.finish_reason == "length"
            if boxed is not None:
                n_boxed += 1
                n_correct += bool(check_equal(boxed, row["answer"]))
        n = len(o.outputs)
        records.append({
            "idx": i,
            "problem": row["problem"],
            "answer": row["answer"],
            "domain": row["domain"],
            "llama8b_solve_rate": row["llama8b_solve_rate"],
            "osim_solve_rate": n_correct / n,
            "osim_boxed_rate": n_boxed / n,
            "osim_trunc_rate": n_trunc / n,
        })

    keep = [r for r in records if KEEP_LO < r["osim_solve_rate"] < KEEP_HI]
    at_floor = [r for r in records if r["osim_solve_rate"] == 0.0]
    at_ceil = [r for r in records if r["osim_solve_rate"] >= KEEP_HI]

    hist = collections.Counter(
        min(int(r["osim_solve_rate"] * 10) / 10, 0.9) for r in records
    )

    summary = {
        "n_scored": len(records),
        "n_keep": len(keep),
        "keep_frac": round(len(keep) / max(1, len(records)), 4),
        "n_at_floor_rate0": len(at_floor),
        "n_at_ceiling_ge0.6": len(at_ceil),
        "mean_osim_solve_rate": round(
            sum(r["osim_solve_rate"] for r in records) / max(1, len(records)), 4),
        "mean_llama8b_solve_rate": round(
            sum(r["llama8b_solve_rate"] for r in records) / max(1, len(records)), 4),
        "mean_boxed_rate": round(
            sum(r["osim_boxed_rate"] for r in records) / max(1, len(records)), 4),
        "mean_trunc_rate": round(
            sum(r["osim_trunc_rate"] for r in records) / max(1, len(records)), 4),
        "histogram": {f"{k:.1f}": hist[k] for k in sorted(hist)},
        "keep_rule": f"{KEEP_LO} < osim_solve_rate < {KEEP_HI}",
        "config": vars(args),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if k not in ("histogram", "config"):
            print(f"  {k}: {v}")
    print("\n  osim_solve_rate histogram:")
    for k, v in summary["histogram"].items():
        print(f"    [{k}, +0.1): {v}")
    print(f"\nWrote {args.out}")
    if summary["keep_frac"] < 0.25:
        print("WARNING: surviving pool is small. Either loosen KEEP_HI or draw "
              "candidates from unfiltered Big-Math rather than this subset.")


if __name__ == "__main__":
    main()
