"""Build the OSIM-calibrated training/eval dataset from OpenR1-Math-220k.

Big-Math-RL-Verified-Filtered is unusable with an OSIM student: it was
difficulty-filtered against Llama-3.1-8B, and OSIM solves it at 0.747, leaving
only 13.5% inside `0 < rate < 0.6` (logs/osim_solve_rates.json). OpenR1 is
harder (OSIM 0.375), 20x larger, and 99% of its rows carry a worked solution --
which the teacher needs, since teacher_prompt.txt is rendered without the
answer and the teacher would otherwise have to out-solve a stronger student.

Three filters, applied in cost order (cheapest first):

  1. SOLUTION PRESENT      -- required for the reference-solution teacher prompt.
  2. SOLUTION LENGTH       -- <= --max-solution-tokens. A few-turn dialogue with
     512-token turns cannot scaffold a problem whose reference solution runs to
     800 tokens; those need long single-shot reasoning, not tutoring. OpenR1's
     median is 289 tokens and p90 is 765, so this trims the long tail.
  3. OSIM SOLVE RATE       -- `0 < rate < 0.6`, the same rule TutorRL's own
     dataset satisfies (its llama8b_solve_rate band is exactly 1/64..38/64).
     rate 0 problems give no measurable delta; rate >= 0.6 leaves no headroom.

Stages are separate so the expensive one is resumable and sharded:

  prefilter : CPU only. Loads OpenR1, applies filters 1-2, writes a candidate list.
  score     : GPU. Scores one shard of candidates with OSIM, checkpointing every
              --chunk problems so a crash costs one chunk, not the whole run.
  assemble  : CPU only. Merges shards, applies filter 3, splits train/test with
              problem-text disjointness enforced, writes the dataset.

Note on splits: rd211's own splits are NOT disjoint (train and test share 17
problems), so this script deduplicates on problem text before splitting.

Usage:
  python utils/build_osim_dataset.py --stage prefilter --limit 30000
  CUDA_VISIBLE_DEVICES=k python utils/build_osim_dataset.py --stage score \\
      --shard k --num-shards 6
  python utils/build_osim_dataset.py --stage assemble --test-size 500
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import extract_answer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "open-r1/OpenR1-Math-220k"
OSIM_MODEL = "cmu-lti/osim-8b"
TOKENIZER = "Qwen/Qwen3-8B"
PROMPT_PATH = os.path.join(ROOT, "prompt_templates",
                           "student_initial_attempt_prompt.txt")

CAND_PATH = "logs/openr1_candidates.json"
SCORE_GLOB = "logs/openr1_scores_shard*.json"

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
KEEP_LO, KEEP_HI = 0.0, 0.6


def _norm(s):
    s = str(s).strip().strip("$").replace(" ", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac")
    return s.rstrip(".").lower()


def make_verifier():
    try:
        from math_verify import parse as mv_parse, verify as mv_verify
        return lambda p, g: bool(mv_verify(mv_parse(str(g)), mv_parse(str(p))))
    except Exception:
        return None


def is_correct(pred, gold, verify_fn):
    if pred is None:
        return False
    if _norm(pred) == _norm(gold):
        return True
    if verify_fn is None:
        return False
    try:
        return verify_fn(pred, gold)
    except Exception:
        return False


# ---------------------------------------------------------------- prefilter
def stage_prefilter(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ds = load_dataset(DATASET, split="train")
    print(f"loaded {len(ds)} rows", flush=True)

    seen, out = set(), []
    n_nosol = n_long = n_dup = n_noans = 0
    for row in ds:
        prob, ans, sol = row.get("problem"), row.get("answer"), row.get("solution")
        if not prob or not str(ans or "").strip():
            n_noans += 1
            continue
        if not sol or len(str(sol).strip()) < 10:
            n_nosol += 1
            continue
        key = str(prob).strip()
        if key in seen:
            n_dup += 1
            continue
        n_sol_tok = len(tok.encode(str(sol)))
        if n_sol_tok > args.max_solution_tokens:
            n_long += 1
            continue
        seen.add(key)
        out.append({"problem": str(prob), "answer": str(ans),
                    "solution": str(sol), "n_solution_tokens": n_sol_tok})
        if args.limit > 0 and len(out) >= args.limit:
            break

    os.makedirs(os.path.dirname(CAND_PATH) or ".", exist_ok=True)
    with open(CAND_PATH, "w") as f:
        json.dump(out, f)
    print(f"kept {len(out)}  (dropped: no-answer {n_noans}, no-solution {n_nosol}, "
          f"solution>{args.max_solution_tokens}tok {n_long}, duplicate {n_dup})")
    print(f"-> {CAND_PATH}")


# -------------------------------------------------------------------- score
def stage_score(args):
    from jinja2 import Template
    from vllm import LLM, SamplingParams

    cands = json.load(open(CAND_PATH))
    mine = cands[args.shard::args.num_shards]
    out_path = f"logs/openr1_scores_shard{args.shard}.json"

    done = {}
    if os.path.exists(out_path) and not args.restart:
        done = {d["problem"]: d for d in json.load(open(out_path))}
        print(f"resuming: {len(done)} already scored", flush=True)
    todo = [c for c in mine if c["problem"] not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} assigned, "
          f"{len(todo)} to score", flush=True)
    if not todo:
        return

    tpl = Template(open(PROMPT_PATH).read())
    verify_fn = make_verifier()
    llm = LLM(model=OSIM_MODEL, gpu_memory_utilization=args.gpu_util,
              max_model_len=8192, enforce_eager=True)
    sp = SamplingParams(n=args.samples, temperature=args.temperature,
                        top_p=1.0, max_tokens=args.max_tokens)

    results = list(done.values())
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        prompts = [[{"role": "user", "content": tpl.render(problem=c["problem"])}]
                   for c in chunk]
        try:
            outs = llm.chat(prompts, sp,
                            chat_template_kwargs={"enable_thinking": False})
        except Exception:
            outs = llm.chat(prompts, sp)
        for c, o in zip(chunk, outs):
            n_ok = n_boxed = 0
            for comp in o.outputs:
                text = _THINK_RE.sub("", comp.text).strip()
                boxed = extract_answer(text)
                if boxed is not None:
                    n_boxed += 1
                    n_ok += is_correct(boxed, c["answer"], verify_fn)
            n = len(o.outputs)
            results.append({**c, "osim_solve_rate": n_ok / n,
                            "osim_boxed_rate": n_boxed / n})
        # Checkpoint after every chunk: a crash costs one chunk, not the run.
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  shard {args.shard}: {len(results)}/{len(mine)} scored", flush=True)
    print(f"-> {out_path}")


# ----------------------------------------------------------------- assemble
def stage_assemble(args):
    import random

    files = sorted(glob.glob(SCORE_GLOB))
    if not files:
        sys.exit(f"no score shards matching {SCORE_GLOB}")
    rows = []
    for f in files:
        rows.extend(json.load(open(f)))
    print(f"merged {len(rows)} scored problems from {len(files)} shards")

    # Deduplicate on problem text: rd211's own splits overlap by 17 problems,
    # so never assume the source is clean.
    seen, uniq = set(), []
    for r in rows:
        k = r["problem"].strip()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    print(f"  unique problems: {len(uniq)}")

    keep = [r for r in uniq if KEEP_LO < r["osim_solve_rate"] < KEEP_HI]
    at_floor = sum(1 for r in uniq if r["osim_solve_rate"] == 0)
    at_ceil = sum(1 for r in uniq if r["osim_solve_rate"] >= KEEP_HI)
    print(f"  rate==0 (unteachable floor): {at_floor} ({at_floor/len(uniq):.1%})")
    print(f"  rate>=0.6 (no headroom):     {at_ceil} ({at_ceil/len(uniq):.1%})")
    print(f"  KEPT 0<rate<0.6:             {len(keep)} ({len(keep)/len(uniq):.1%})")

    if len(keep) < args.test_size * 2:
        sys.exit(f"only {len(keep)} problems survive; too few to split")

    rng = random.Random(args.seed)
    rng.shuffle(keep)
    test, train = keep[:args.test_size], keep[args.test_size:]
    assert not (set(r["problem"] for r in train) & set(r["problem"] for r in test))

    os.makedirs(args.out, exist_ok=True)
    for name, split in (("train", train), ("test", test)):
        p = os.path.join(args.out, f"{name}.jsonl")
        with open(p, "w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
        rates = [r["osim_solve_rate"] for r in split]
        toks = [r["n_solution_tokens"] for r in split]
        print(f"{name}: {len(split)} -> {p}")
        print(f"   mean osim_solve_rate {sum(rates)/len(rates):.3f}, "
              f"mean solution tokens {sum(toks)/len(toks):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prefilter", "score", "assemble"])
    ap.add_argument("--limit", type=int, default=30000,
                    help="prefilter: max candidates to keep")
    ap.add_argument("--max-solution-tokens", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--test-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/openr1_osim")
    args = ap.parse_args()
    {"prefilter": stage_prefilter, "score": stage_score,
     "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
