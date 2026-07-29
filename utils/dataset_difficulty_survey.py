"""Survey candidate math datasets for a usable difficulty band, for BOTH the
student simulator and the teacher policy.

Two constraints have to hold at once:

1. STUDENT HEADROOM. Problems OSIM already solves leave nothing for a tutor to
   teach, and problems it never solves leave nothing to measure. Big-Math-
   Filtered fails this badly: OSIM's mean solve rate there is 0.747 and only
   13.5% of it survives `0 < rate < 0.6` (logs/osim_solve_rates.json).

2. TEACHER COMPETENCE. The teacher prompt passes only the PROBLEM, never the
   ground-truth answer (src/classroom.py renders teacher_prompt.txt with
   student_name/problem/include_thinking only). So the teacher has to solve the
   problem itself to tutor it. If the student model is stronger at math than the
   teacher, the tutoring relationship is inverted and neither the learning
   reward nor Delta solve rate means what it is supposed to. Qwen2.5-7B-Instruct
   (train policy) and TutorRL-7B (baseline) are both weaker families than
   OSIM's Qwen3-8B base, so this must be measured, not assumed.

Run the same script once per model and compare: a usable dataset is one where
teacher solve rate comfortably exceeds student solve rate AND the student's
rate leaves headroom.

Answer checking uses math_verify (LaTeX-aware) with a normalized string
fallback. The repo's own check_equal is bare string equality with math_verify
commented out, which would systematically understate solve rates on datasets
whose answers are LaTeX expressions and make them look artificially hard.

Usage:
    CUDA_VISIBLE_DEVICES=0 python utils/dataset_difficulty_survey.py \
        --model cmu-lti/osim-8b --limit 300 --out logs/survey_osim.json
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

from src.utils.utils import extract_answer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(ROOT, "prompt_templates",
                           "student_initial_attempt_prompt.txt")

# (hf_name, config, split, problem_col, answer_col, has_worked_solution)
# has_worked_solution matters because the teacher currently gets no ground
# truth; a reference solution column is what would let us give it one.
DATASETS = {
    "bigmath":     ("rd211/Big-Math-RL-Verified-Filtered", None, "train",
                    "problem", "answer", False),
    "math500":     ("HuggingFaceH4/MATH-500", None, "test",
                    "problem", "answer", True),
    "omni_math":   ("KbsdJames/Omni-MATH", None, "test",
                    "problem", "answer", True),
    "deepscaler":  ("agentica-org/DeepScaleR-Preview-Dataset", None, "train",
                    "problem", "answer", True),
    "openr1":      ("open-r1/OpenR1-Math-220k", None, "train",
                    "problem", "answer", True),
    "aime_83_24":  ("di-zhang-fdu/AIME_1983_2024", None, "train",
                    "Question", "Answer", False),
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
KEEP_LO, KEEP_HI = 0.0, 0.6


def _norm(s):
    s = str(s).strip().strip("$").replace(" ", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac")
    s = s.rstrip(".")
    return s.lower()


def is_correct(pred, gold, verify_fn):
    if pred is None:
        return False
    if _norm(pred) == _norm(gold):
        return True
    if verify_fn is not None:
        try:
            return bool(verify_fn(pred, gold))
        except Exception:
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        from math_verify import parse as mv_parse, verify as mv_verify

        def verify_fn(pred, gold):
            return mv_verify(mv_parse(str(gold)), mv_parse(str(pred)))
    except Exception as e:
        print(f"math_verify unavailable ({e}); string-only comparison")
        verify_fn = None

    with open(PROMPT_PATH) as f:
        tpl = Template(f.read())

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    loaded = {}
    for key in wanted:
        name, cfg, split, pcol, acol, has_sol = DATASETS[key]
        try:
            ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
            ds = ds.shuffle(seed=args.seed).select(range(min(args.limit, len(ds))))
            rows = [(str(p), str(a)) for p, a in zip(ds[pcol], ds[acol])
                    if p is not None and a is not None and str(a).strip()]
            loaded[key] = (rows, has_sol)
            print(f"loaded {key}: {len(rows)} problems", flush=True)
        except Exception as e:
            print(f"SKIP {key}: {repr(e)[:120]}", flush=True)

    prompts, meta = [], []
    for key, (rows, _) in loaded.items():
        for i, (p, a) in enumerate(rows):
            prompts.append([{"role": "user", "content": tpl.render(problem=p)}])
            meta.append({"ds": key, "i": i, "answer": a})

    print(f"total {len(prompts)} problems x {args.samples} samples", flush=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_model_len, enforce_eager=True)
    sp = SamplingParams(n=args.samples, temperature=args.temperature,
                        top_p=1.0, max_tokens=args.max_tokens)
    try:
        outs = llm.chat(prompts, sp, chat_template_kwargs={"enable_thinking": False})
    except Exception:
        outs = llm.chat(prompts, sp)

    per_ds = collections.defaultdict(list)
    for m, o in zip(meta, outs):
        n_ok = n_boxed = 0
        for comp in o.outputs:
            text = _THINK_RE.sub("", comp.text).strip()
            boxed = extract_answer(text)
            if boxed is not None:
                n_boxed += 1
                n_ok += is_correct(boxed, m["answer"], verify_fn)
        n = len(o.outputs)
        per_ds[m["ds"]].append({"i": m["i"], "rate": n_ok / n, "boxed": n_boxed / n})

    summary = {}
    for key, rs in per_ds.items():
        rates = [r["rate"] for r in rs]
        keep = [r for r in rates if KEEP_LO < r < KEEP_HI]
        summary[key] = {
            "n": len(rs),
            "mean_solve_rate": round(sum(rates) / len(rates), 4),
            "frac_rate_0": round(sum(r == 0 for r in rates) / len(rates), 4),
            "frac_ceiling_ge0.6": round(sum(r >= KEEP_HI for r in rates) / len(rates), 4),
            "keep_frac_0_to_0.6": round(len(keep) / len(rates), 4),
            "mean_boxed_rate": round(sum(r["boxed"] for r in rs) / len(rs), 4),
            "has_worked_solution": loaded[key][1],
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        # Per-problem rates are required for the decisive cross-model metric:
        # the USABLE BAND is problems the student fails but the teacher solves,
        # which cannot be recovered from marginal solve rates alone.
        json.dump({"model": args.model, "summary": summary,
                   "per_problem": {k: v for k, v in per_ds.items()},
                   "config": vars(args)}, f, indent=2)

    print(f"\n=== {args.model} ===")
    print(f"{'dataset':<14}{'n':>5}{'mean':>8}{'r=0':>8}{'ceil':>8}{'keep':>8}{'boxed':>8}{'soln':>6}")
    for k, s in summary.items():
        print(f"{k:<14}{s['n']:>5}{s['mean_solve_rate']:>8}{s['frac_rate_0']:>8}"
              f"{s['frac_ceiling_ge0.6']:>8}{s['keep_frac_0_to_0.6']:>8}"
              f"{s['mean_boxed_rate']:>8}{str(s['has_worked_solution']):>6}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
