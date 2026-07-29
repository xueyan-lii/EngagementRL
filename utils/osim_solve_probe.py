"""OSIM (cmu-lti/osim-8b) solve-rate + volunteering probe.

Two questions this answers, both blockers for switching the RL student
simulator from UserLM-8b to OSIM:

  Q1 (ELICITED) -- can OSIM, when the tutor asks it to, produce a complete
      solution with an extractable \\boxed{} answer?  If yes, the Delta Solve
      Rate probe no longer needs a separate Llama student model: OSIM can be
      asked for its own initial attempt and its own post-dialogue solution,
      which also removes the current probe/simulator mismatch (the Llama probe
      and the UserLM talker are different students).

  Q2 (VOLUNTEERED) -- how often does OSIM write out a full solution *without*
      being asked, i.e. in response to an ordinary Socratic tutor turn?  This
      is the confound for the learning reward: every unprompted solution is
      student-produced, quotable work that satisfies the terminal judge's
      EVIDENCE RULE regardless of whether the tutor taught anything.  A high
      volunteering rate means learning reward has a large tutor-independent
      floor and problem filtering must be recalibrated to OSIM.

Both are measured under the two personas we intend to train against
(passive / engaged, fixed per run -- never varied within a GRPO group), so the
same run also reports whether persona moves solve behaviour at all.

Role mapping (OSIM is the INVERSE of UserLM): system = persona describing the
human; `user` = the tutor; `assistant` = the simulated student's own turns.

Problems are stratified by the dataset's precomputed `llama8b_solve_rate` so
the output is directly usable for difficulty filtering: it gives OSIM's solve
rate as a function of Llama's, which is what decides which Big-Math slice
still leaves room for a tutor to matter.

Usage (from the repo root):
    CUDA_VISIBLE_DEVICES=<free gpu> python utils/osim_solve_probe.py \
        --n-per-bin 40 --samples 4 --out logs/osim_solve_probe.json
"""

import argparse
import collections
import json
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from jinja2 import Template
from vllm import LLM, SamplingParams

from src.utils.utils import check_equal, extract_answer

OSIM_MODEL = "cmu-lti/osim-8b"
DATASET = "rd211/Big-Math-RL-Verified-Filtered"

PROMPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "prompt_templates")

# Difficulty bins over llama8b_solve_rate. The two middle bins are where a
# tutor can plausibly change the outcome; the outer bins are the floor/ceiling
# controls that tell us whether the metric saturates.
BINS = [(0.0, 0.15), (0.15, 0.40), (0.40, 0.70), (0.70, 1.01)]

# A neutral Socratic opening tutor turn that does NOT ask for a full solution.
# Deliberately generic: it must not hint at the method, or "volunteering"
# would just be measuring how strong the hint was.
SOCRATIC_OPENER = Template(
    "Hi! Let's work on this one together:\n\n{{ problem }}\n\n"
    "Before I say anything about how to approach it -- what do you notice about "
    "the problem? What's the first thing you'd try?"
)

# Raw-capability control: no persona, plain instruct framing. Establishes the
# ceiling, so an in-persona shortfall can be read as persona suppression
# rather than inability.
CAPABILITY_SYSTEM = "You are a math student. Solve the problem you are given."
CAPABILITY_USER = Template(
    "Solve this problem. Think step by step but keep it concise. It is essential "
    "you include the final answer in \\boxed{}.\n\nHere is the problem:\n{{ problem }}"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)

# Heuristic "this turn contains real mathematical work" detector, used only for
# the volunteering rate. Conservative and reported alongside raw transcripts --
# treat it as a screening number, not ground truth.
_WORK_RE = re.compile(r"=\s*[-+]?\d|\\frac|\\boxed|\d\s*[+\-*/^]\s*\d")


def read_template(name):
    with open(os.path.join(PROMPTS, name)) as f:
        return Template(f.read())


def build_conditions(problem, personas, tpl_initial):
    """Returns {condition_name: chat messages} for one problem."""
    conds = {
        "capability_ceiling": [
            {"role": "system", "content": CAPABILITY_SYSTEM},
            {"role": "user", "content": CAPABILITY_USER.render(problem=problem)},
        ]
    }
    for pname, ptext in personas.items():
        # Q1: tutor explicitly asks for a full attempt.
        conds[f"elicited_{pname}"] = [
            {"role": "system", "content": ptext},
            {"role": "user", "content": tpl_initial.render(problem=problem)},
        ]
        # Q2: ordinary Socratic opener, no request to solve.
        conds[f"volunteer_{pname}"] = [
            {"role": "system", "content": ptext},
            {"role": "user", "content": SOCRATIC_OPENER.render(problem=problem)},
        ]
    return conds


def stratified_sample(ds, n_per_bin, seed):
    rng = random.Random(seed)
    by_bin = collections.defaultdict(list)
    for i, sr in enumerate(ds["llama8b_solve_rate"]):
        for lo, hi in BINS:
            if lo <= sr < hi:
                by_bin[(lo, hi)].append(i)
                break
    picked = []
    for b in BINS:
        idxs = by_bin.get(b, [])
        if not idxs:
            print(f"WARNING: difficulty bin {b} is empty; skipping")
            continue
        if len(idxs) < n_per_bin:
            print(f"WARNING: bin {b} has only {len(idxs)} problems "
                  f"(asked for {n_per_bin}); using all of them")
        picked += [(b, i) for i in rng.sample(idxs, min(n_per_bin, len(idxs)))]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-bin", type=int, default=40)
    ap.add_argument("--samples", type=int, default=4,
                    help="samples per (problem, condition); >1 gives a per-problem "
                         "solve rate rather than a single Bernoulli draw")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="matches the rollout sampling params, not a decoding "
                         "optimum -- this is the distribution RL will actually see")
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="logs/osim_solve_probe.json")
    args = ap.parse_args()

    personas = {
        "passive": read_template("personas/osim_passive.txt").render(),
        "engaged": read_template("personas/osim_engaged.txt").render(),
    }
    tpl_initial = read_template("osim_initial_attempt_prompt.txt")

    ds = load_dataset(DATASET, split=args.split)
    picked = stratified_sample(ds, args.n_per_bin, args.seed)
    print(f"Sampled {len(picked)} problems across {len(BINS)} difficulty bins")

    # Flatten every (problem, condition, sample) into one batch.
    prompts, meta = [], []
    for bin_key, idx in picked:
        problem, answer = ds[idx]["problem"], ds[idx]["answer"]
        for cond, msgs in build_conditions(problem, personas, tpl_initial).items():
            for s in range(args.samples):
                prompts.append(msgs)
                meta.append({"bin": f"{bin_key[0]:.2f}-{bin_key[1]:.2f}",
                             "idx": idx, "cond": cond, "sample": s,
                             "answer": answer, "llama_sr": ds[idx]["llama8b_solve_rate"]})

    llm = LLM(
        model=OSIM_MODEL,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=4096,
        enforce_eager=True,
    )
    # NOTE: no `seed` here. A fixed seed makes vLLM sampling deterministic per
    # identical prompt, so the `--samples` draws of one (problem, condition)
    # collapse to the same string and the effective n is silently divided by
    # `--samples`. args.seed governs dataset selection only.
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens)

    # OSIM is post-trained from Qwen3-8B, whose stock template exposes a
    # thinking mode. A user simulator must not emit reasoning blocks as if they
    # were student speech, so disable it -- and strip any <think> block anyway
    # in case this checkpoint's template ignores the flag.
    try:
        outs = llm.chat(prompts, sp, chat_template_kwargs={"enable_thinking": False})
    except Exception as e:
        print(f"enable_thinking kwarg rejected ({e}); retrying without it")
        outs = llm.chat(prompts, sp)

    records = []
    for m, o in zip(meta, outs):
        out0 = o.outputs[0]
        text = _THINK_RE.sub("", out0.text).strip()
        boxed = extract_answer(text)
        records.append({
            **m,
            "text": text,
            "boxed": boxed,
            "correct": bool(boxed is not None and check_equal(boxed, m["answer"])),
            "has_work": bool(_WORK_RE.search(text)),
            "n_chars": len(text),
            # Distinguishes "hit the token cap before reaching \boxed{}" from
            # "declined to attempt the problem" -- these have opposite fixes.
            "truncated": out0.finish_reason == "length",
        })

    # ---- aggregate ----
    def agg(rs):
        if not rs:
            return None
        return {
            "n": len(rs),
            "boxed_rate": round(sum(r["boxed"] is not None for r in rs) / len(rs), 3),
            "correct_rate": round(sum(r["correct"] for r in rs) / len(rs), 3),
            "has_work_rate": round(sum(r["has_work"] for r in rs) / len(rs), 3),
            "trunc_rate": round(sum(r["truncated"] for r in rs) / len(rs), 3),
            "median_chars": int(statistics.median(r["n_chars"] for r in rs)),
        }

    by_cond = collections.defaultdict(list)
    by_cond_bin = collections.defaultdict(list)
    for r in records:
        by_cond[r["cond"]].append(r)
        by_cond_bin[(r["cond"], r["bin"])].append(r)

    summary = {
        "overall": {c: agg(rs) for c, rs in sorted(by_cond.items())},
        "by_difficulty": {
            f"{c} | llama_sr {b}": agg(rs)
            for (c, b), rs in sorted(by_cond_bin.items())
        },
        "config": vars(args),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print("\n=== OVERALL ===")
    print(f"{'condition':<28}{'n':>6}{'boxed':>8}{'correct':>9}{'work':>8}"
          f"{'trunc':>8}{'chars':>8}")
    for c, a in summary["overall"].items():
        print(f"{c:<28}{a['n']:>6}{a['boxed_rate']:>8}{a['correct_rate']:>9}"
              f"{a['has_work_rate']:>8}{a['trunc_rate']:>8}{a['median_chars']:>8}")
    print("\n=== BY DIFFICULTY (llama8b_solve_rate bin) ===")
    for k, a in summary["by_difficulty"].items():
        print(f"{k:<48}{a['n']:>6}{a['boxed_rate']:>8}{a['correct_rate']:>9}")
    print(f"\nFull records -> {args.out}")
    print("Read ~20 `volunteer_passive` transcripts by hand: has_work is a "
          "regex screen, not a judgement of whether a real solution was given.")


if __name__ == "__main__":
    main()
