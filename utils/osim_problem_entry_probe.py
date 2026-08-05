"""How should the PROBLEM STATEMENT reach OSIM?

Under UserLM the problem arrived via a synthetic student-first opening turn,
which existed to keep UserLM inside its student-first training distribution.
OSIM does not need that for format reasons -- a GUIDED conversation is already
teacher-first, which matches OSIM's natural user->assistant order -- but the
synthetic opening was ALSO how the problem text got in front of the student.
Removing it without replacing it risks a student that does not know the task.

Four ways the problem can enter, tested against the REAL teacher policy
(Qwen2.5-7B-Instruct rendering prompt_templates/teacher_prompt.txt, which holds
the problem in the TEACHER's system prompt and never requires it to be
restated):

  A system_only       problem appended to OSIM's persona/system slot
  B tutor_only        problem nowhere on the student side; relies entirely on
                      the teacher choosing to restate it
  C tutor_explicit    harness prefixes the problem onto the teacher's first
                      visible turn ("tutor says it")
  D synthetic_opening the UserLM port: a fabricated student turn stating the
                      problem, then the teacher's turn

Measured over two real dialogue turns:
  - confusion:  student says it cannot see / was not given a problem
  - grounding:  numbers in the student's turn that actually occur in the
                problem (hallucinated numbers = the student invented a task)
  - drift:      student slips into tutor voice
  - length

Both models are resident on one GPU at reduced utilization so an actual
two-turn exchange can be run rather than a single scripted turn.

Usage:
    CUDA_VISIBLE_DEVICES=0 python utils/osim_problem_entry_probe.py --n 40
"""

import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from jinja2 import Template
from vllm import LLM, SamplingParams

OSIM_MODEL = "cmu-lti/osim-8b"
TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATASET = "rd211/Big-Math-RL-Verified-Filtered"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Explicit "I was not given a problem" signals. These are the decisive failure:
# a student that does not know the task cannot be tutored at all.
_CONFUSION_RE = re.compile(
    r"(what (is |was )?the problem"
    r"|which problem"
    r"|i don'?t (see|have) (the|a|any) problem"
    r"|you haven'?t (given|shared|sent|posted)"
    r"|didn'?t (give|share|send|post) me"
    r"|no problem (was|has been)"
    r"|(can|could) you (share|send|post|give me) the problem"
    r"|what (are we|am i) (working on|solving)"
    r"|there'?s no problem"
    r"|i can'?t see (the|any) problem)",
    re.IGNORECASE,
)


def render_teacher_system(problem, use_thinking=True):
    with open(os.path.join(ROOT, "prompt_templates", "teacher_prompt.txt")) as f:
        return Template(f.read()).render(
            problem=problem, student_name=None, include_thinking=use_thinking
        )


def grounding(student_text, problem):
    """Share of numbers in the student's turn that occur in the problem.
    Low share + numbers present = the student invented its own task."""
    s_nums = _NUM_RE.findall(student_text)
    if not s_nums:
        return None  # no numeric claim to check
    p_nums = set(_NUM_RE.findall(problem))
    return sum(n in p_nums for n in s_nums) / len(s_nums)


def build_student_msgs(cond, persona, problem, convo):
    """convo: [(role, text)] with role in {'teacher','student'} in order."""
    system = persona
    if cond == "A_system_only":
        system = f"{persona}\n\nThe problem you are working on:\n\n{problem}"
    msgs = [{"role": "system", "content": system}]
    if cond == "D_synthetic_opening":
        msgs.append({
            "role": "assistant",
            "content": f"Here's a problem I'm trying to solve:\n\n{problem}"
                       "\n\nCan you help me?",
        })
    for i, (role, text) in enumerate(convo):
        if cond == "C_tutor_explicit" and role == "teacher" and i == 0:
            text = f"Let's work on this problem:\n\n{problem}\n\n{text}"
        # OSIM's simulated human generates the `assistant` role; the tutor is
        # the `user`. This is the inverse of the UserLM mapping.
        msgs.append({"role": "user" if role == "teacher" else "assistant",
                     "content": text})
    return msgs


def build_teacher_msgs(problem, convo):
    msgs = [{"role": "system", "content": render_teacher_system(problem)}]
    for role, text in convo:
        msgs.append({"role": "assistant" if role == "teacher" else "user",
                     "content": text})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--turns", type=int, default=2, help="student turns per dialogue")
    ap.add_argument("--gpu-util", type=float, default=0.40,
                    help="per-model; two models share one GPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--persona", default="osim_disengaged",
                    help="persona file stem under prompt_templates/personas/")
    ap.add_argument("--conds", default="A_system_only",
                    help="comma-separated subset of conditions to run")
    ap.add_argument("--out", default="logs/osim_problem_entry.json")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "prompt_templates", "personas",
                           f"{args.persona}.txt")) as f:
        persona = f.read().strip()

    ds = load_dataset(DATASET, split="train").shuffle(seed=args.seed)
    ds = ds.select(range(args.n))
    problems = ds["problem"]

    teacher = LLM(model=TEACHER_MODEL, gpu_memory_utilization=args.gpu_util,
                  max_model_len=8192, enforce_eager=True)
    sp_teacher = SamplingParams(temperature=1.0, top_k=50, top_p=1.0,
                                max_tokens=512,
                                stop=["\nUser:", "\nStudent:", "\nHuman:"])

    osim = LLM(model=OSIM_MODEL, gpu_memory_utilization=args.gpu_util,
               max_model_len=8192, enforce_eager=True)
    sp_osim = SamplingParams(temperature=1.0, top_p=0.8, max_tokens=512)

    def osim_chat(msgs_list):
        try:
            return osim.chat(msgs_list, sp_osim,
                             chat_template_kwargs={"enable_thinking": False})
        except Exception:
            return osim.chat(msgs_list, sp_osim)

    CONDS = [c.strip() for c in args.conds.split(",") if c.strip()]

    # Teacher's opening turn depends only on the problem, so it is generated
    # once and shared across conditions -- this isolates the entry mechanism as
    # the sole difference.
    print("Generating teacher opening turns...", flush=True)
    t_outs = teacher.chat([build_teacher_msgs(p, []) for p in problems], sp_teacher)
    opening = [_THINK_RE.sub("", o.outputs[0].text).replace(
        "<end_of_conversation>", "").strip() for o in t_outs]

    convos = {c: [[("teacher", opening[i])] for i in range(args.n)] for c in CONDS}
    records = []

    for turn in range(args.turns):
        print(f"--- student turn {turn + 1} ---", flush=True)
        for cond in CONDS:
            msgs = [build_student_msgs(cond, persona, problems[i], convos[cond][i])
                    for i in range(args.n)]
            outs = osim_chat(msgs)
            for i, o in enumerate(outs):
                text = _THINK_RE.sub("", o.outputs[0].text).strip()
                convos[cond][i].append(("student", text))
                records.append({
                    "cond": cond, "turn": turn + 1, "idx": i,
                    "text": text,
                    "confused": bool(_CONFUSION_RE.search(text)),
                    "grounding": grounding(text, problems[i]),
                    "n_chars": len(text),
                })

        if turn < args.turns - 1:
            print(f"--- teacher turn {turn + 2} ---", flush=True)
            for cond in CONDS:
                msgs = [build_teacher_msgs(problems[i], convos[cond][i])
                        for i in range(args.n)]
                outs = teacher.chat(msgs, sp_teacher)
                for i, o in enumerate(outs):
                    t = _THINK_RE.sub("", o.outputs[0].text).replace(
                        "<end_of_conversation>", "").strip()
                    convos[cond][i].append(("teacher", t))

    # Does the teacher spontaneously restate the problem? This is what
    # condition B silently depends on.
    restate = []
    for i, p in enumerate(problems):
        p_nums = set(_NUM_RE.findall(p))
        o_nums = set(_NUM_RE.findall(opening[i]))
        restate.append(len(p_nums & o_nums) / len(p_nums) if p_nums else None)
    restate_vals = [r for r in restate if r is not None]

    def agg(rs):
        gs = [r["grounding"] for r in rs if r["grounding"] is not None]
        return {
            "n": len(rs),
            "confusion_rate": round(sum(r["confused"] for r in rs) / len(rs), 3),
            "no_numbers_rate": round(
                sum(r["grounding"] is None for r in rs) / len(rs), 3),
            "mean_grounding": round(sum(gs) / len(gs), 3) if gs else None,
            "fully_grounded_rate": round(
                sum(g == 1.0 for g in gs) / len(gs), 3) if gs else None,
            "median_chars": int(statistics.median(r["n_chars"] for r in rs)),
        }

    by = collections.defaultdict(list)
    for r in records:
        by[(r["cond"], r["turn"])].append(r)
        by[(r["cond"], "all")].append(r)

    summary = {
        "teacher_restates_problem_numbers_mean": round(
            sum(restate_vals) / len(restate_vals), 3) if restate_vals else None,
        "teacher_fully_restates_rate": round(
            sum(v == 1.0 for v in restate_vals) / len(restate_vals), 3
        ) if restate_vals else None,
        "by_condition": {f"{c} | turn {t}": agg(rs)
                         for (c, t), rs in sorted(by.items(), key=lambda kv: str(kv[0]))},
        "config": vars(args),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records,
                   "openings": opening, "problems": list(problems)}, f, indent=2)

    print("\n=== TEACHER OPENING ===")
    print(f"  mean share of problem numbers restated: "
          f"{summary['teacher_restates_problem_numbers_mean']}")
    print(f"  fully restates: {summary['teacher_fully_restates_rate']}")
    print("\n=== STUDENT TURNS ===")
    print(f"{'condition | turn':<34}{'n':>5}{'confus':>8}{'ground':>8}"
          f"{'full':>7}{'nonum':>7}{'chars':>7}")
    for k, a in summary["by_condition"].items():
        print(f"{k:<34}{a['n']:>5}{a['confusion_rate']:>8}"
              f"{str(a['mean_grounding']):>8}{str(a['fully_grounded_rate']):>7}"
              f"{a['no_numbers_rate']:>7}{a['median_chars']:>7}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
