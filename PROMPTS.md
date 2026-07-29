# All prompts used in the engagement-conditioned RL experiments

Single reference for every prompt in the training + eval pipelines.
**The code is the source of truth** — each section links the file that is
actually loaded at runtime; update there, then mirror here.

| # | Prompt | Used by | Source of truth |
|---|--------|---------|-----------------|
| 1 | Teacher (tutor policy) system prompt | rollouts, train + eval | `prompt_templates/teacher_prompt.txt` |
| 2 | UserLM intent + synthetic opening | student simulator view | `train_engagement_classroom.py` (`INTENT`, `_opening_turn`) |
| 3 | Per-turn engagement judge (+ learning-evidence, role-drift) | training reward | `judge_rewards.py::ENGAGEMENT_SYSTEM_PROMPT` |
| 4 | Terminal learning-outcome judge | training reward | `judge_rewards.py::LEARNING_SYSTEM_PROMPT` |
| 5 | Teacher anti-impersonation stop strings | rollouts (not a prompt, but prompt-adjacent) | `train_engagement_classroom.py` (`sampling_params_teacher`) |
| 6 | Leakage judge (binary OK/REJECT) | eval only | `prompt_templates/judges/does_not_leak_answer.txt` |
| 7 | Pedagogy judge (binary OK/REJECT) | eval only | `prompt_templates/judges/follows_pedagogical_values.txt` |
| 8 | Knowledge-probe prompts (pre/post solve rate) | eval only | `prompt_templates/student_attempt_prompt.txt`, `student_final_prompt.txt`, `initial_attempt_wrapper_prompt.txt` |
| 9 | Persona student (TutorRL's always-engaged baseline student) | baseline eval only | `prompt_templates/personas/simple_student.txt` |
| 10 | Paper-appendix engagement judge (3-dim, dataset study) | §motivation-judge eval | `../engagement_eval/judge_prompt.py` |

Items 1, 6–9 are inherited **verbatim from TutorRL** (comparability); items
2–5, 10 are ours.

---

## 1. Teacher system prompt (`prompt_templates/teacher_prompt.txt`)

Jinja template; rendered with `student_name`, `problem`,
`include_thinking=generation.use_thinking` (true in our runs). Identical to
TutorRL's training prompt — deliberate, for the controlled comparison.

```jinja
{% if student_name %}
You are tasked with being a teacher and helping a student named {{ student_name }} with a math problem.
{% else %}
You are tasked with being a teacher and helping a student with a math problem.
{% endif %}

You must not reveal the answer to the problem to the student at any point in time.
Your task is to guide the student to have a complete understanding of the problem.
Even if the student is already able to solve the problem, you should help them understand and improve the solution so that they get as high of a grade as possible.

If possible, do not respond with overly long responses to the student.

{% if include_thinking %}
In order to be able to think of a good hint or approach for the student without revealing steps of the final solution, you can wrap your internal reasoning like this:
<think>
</think>
[... example of think-tag usage ...]
{% endif %}

You can end a conversation by writing <end_of_conversation>, please try to end conversations as soon as they are finished instead of prolonging them if not needed. But do not end them prematurely either.

Here is the math problem:
{{ problem }}
```

## 2. UserLM student-simulator view (`train_engagement_classroom.py`)

UserLM never sees the teacher's system prompt. Its message list is:

- **system (intent):** `You are a student trying to solve a math problem.`
- **synthetic first user turn (delivers the problem, WildChat-style):**
  `Here's a problem I'm trying to solve:\n\n{problem}\n\nCan you help me?`
- then alternating tutor (assistant) / student (user) turns, with tutor
  `<think>...</think>` blocks and `<end_of_conversation>` markers stripped
  (`engagement_classroom._clean_for_userlm`).

Validated in the baseline eval (Appendix app:baseline): problem-in-intent is
OOD for UserLM; this student-first opening is in-distribution.

## 3. Per-turn engagement judge (`judge_rewards.py::ENGAGEMENT_SYSTEM_PROMPT`)

gpt-oss-20b, temp 1.0, reasoning_effort low, one call per non-empty student
turn (blank turns auto-score 0 without a call). The three engagement
dimensions are verbatim the paper-appendix rubric; `learning_evidence` and
`role_drift` are training additions.

- **BEHAVIORAL (0-4):** 4 drives interaction forward … 2 passive compliance
  ("ok","sure") … 0 refuses/off-task/ends conversation.
- **AFFECTIVE (0-4):** 4 curiosity … 2 flat … 1 negative-but-trying
  (frustration+effort) … 0 checked out.
- **COGNITIVE (0-4, ICAP):** 4 interactive co-construction … 3 constructive
  new reasoning … 2 active manipulation … 1 passive acknowledgment … 0 none.
- **LEARNING EVIDENCE (0-4):** new, correct, *student-produced* reasoning in
  THIS turn; tutor-restated content caps at 1; requires a verbatim quote as
  evidence, else 0.
- **ROLE CHECK:** `drift=true` if the "student" turn reads as an AI assistant
  tutoring the other party (simulator malfunction). Flagged turns are
  excluded from the engagement mean (neither rewarded nor penalized).

Output: JSON with rationale + score per dimension. Full text in
`judge_rewards.py` (kept verbatim there; it is also Appendix app:judge of the
paper minus the two additions).

User message format (same conventions as `engagement_eval/judge_prompt.py`):
`[DIALOGUE SO FAR]` with `Tutor:`/`Student:` labels, last 10 turns, turns
truncated at 1200 chars, then `[TARGET STUDENT TURN -- rate this]`.

**Reward scalar:** mean over non-drift student turns of
`(behavioral+affective+cognitive)/12` → [0,1]. `learning_evidence` is logged
but not currently in the reward.

## 4. Terminal learning-outcome judge (`judge_rewards.py::LEARNING_SYSTEM_PROMPT`)

One call per dialogue at the end; sees problem + ground-truth answer + full
transcript (`Tutor:`/`Student:` labels, 2000-char turn truncation).

Strict rules: **(1) EVIDENCE RULE** — only student-produced work counts; a
tutor-stated answer is worth nothing (this is where leakage is handled at
train time; see PROPOSALS below). **(2) DELTA RULE** — score change during
the dialogue, not incoming ability. **(3) ROLE RULE** — student turns that
read as AI-assistant role-drift earn no learning credit.

Dimensions (0-4 each, JSON with quote+rationale per dimension):
- **SOLUTION PROGRESS** (0 includes "correct answer present but tutor-stated")
- **UNDERSTANDING DEMONSTRATED** (SOLO-style: rote → procedural → relational → generalizes)
- **MISCONCEPTION REPAIR** (−1 = N/A when the student made no initial error)
- **INDEPENDENCE TRAJECTORY** (scaffolding fading)

**Reward scalar:** mean of applicable dims / 4, then multiplied by the
participation gate `min(1, nonempty_student_turns / 2)` (v2 anti-hack: no
real student participation → no learning credit).

## 5. Teacher anti-impersonation stop strings (v2)

`stop=["\nUser:", "\nStudent:", "\nHuman:", "\nLearner:", "\nuser:", "\nstudent:"]`
on the teacher's SamplingParams — generation halts the moment the policy
starts scripting a fake student line inside its own turn (v1 reward hack).

Related, student-side: UserLM assistant-drift regexes + resampling
(`looks_like_assistant`, MAX_DRIFT_RESAMPLES=2) in
`train_engagement_classroom.py`.

## 6–7. Eval-time binary judges (TutorRL, verbatim)

`does_not_leak_answer.txt`: few-shot prompt, decision **OK/REJECT** (not a
score) — REJECT if the tutor gives the full answer upfront, reveals all key
steps in one message, or walks through the solution without involving the
student. Produces the "Leaked-solution rate" in the baseline table.
`follows_pedagogical_values.txt`: same format for pedagogy adherence.
Both run with the gemma-3-27b judge in `eval.py`/`eval_engagement.py`; **not
used during RL training** (`number_judge_attempts: 0`).

## 8. Knowledge-probe prompts (eval-time Δ solve rate, TutorRL verbatim)

- `student_attempt_prompt.txt`: "act as a math student… solve… final answer
  in \boxed{}" — pre-dialogue attempt.
- `student_final_prompt.txt`: "conversation has ended… create a step by step
  complete solution… \boxed{}" — post-dialogue attempt, conditioned on the
  transcript.
- `initial_attempt_wrapper_prompt.txt`: wraps the pre-attempt when it is
  inserted into a dialogue.

## 9. Persona student (baseline's always-engaged student, TutorRL verbatim)

`personas/simple_student.txt` — Llama-3.1-8B roleplaying a cooperative
student ("collaborate with the teacher… you will be tested"). Used only in
the TutorRL×Llama baseline condition, never in our training.

## 10. Paper-appendix engagement judge (`engagement_eval/judge_prompt.py`)

The original 3-dimension rubric used for the 7-corpus real-vs-LLM study
(§motivation-judge, Appendix app:judge). Identical to item 3 minus
`learning_evidence`/`role_drift`. Keep this one frozen — its scores are
already in the paper.

---

# OSIM migration: which prompts are actually in play

Orientation for the OdysSim (`cmu-lti/osim-8b`) student simulator. Most prompts
carry over unchanged from the UserLM setup; the table says what each one does
and whether OSIM touches it.

## The RL training loop needs exactly five prompts

| slot | file / symbol | changes for OSIM? |
|------|---------------|-------------------|
| Tutor policy (system) | `prompt_templates/teacher_prompt.txt` | **No** — unchanged, it is the thing being trained |
| Student persona (system, OSIM side) | `prompt_templates/personas/osim_passive.txt` | **Yes, new** — replaces UserLM's one-line `INTENT` |
| Per-turn engagement judge | `judge_rewards.py::ENGAGEMENT_SYSTEM_PROMPT` | Review only (see below) |
| Terminal learning judge | `judge_rewards.py::LEARNING_SYSTEM_PROMPT` | Review only (see below) |
| Teacher stop-strings (v2 anti-hack) | `train_engagement_classroom.py::sampling_params_teacher` | **No** |

Nothing else is loaded during a training rollout.

## Role mapping is inverted vs UserLM

UserLM sat in the `user` role with the policy as `assistant`. OSIM is the
opposite -- it imitates the human but *generates* the `assistant` role:

    system    = the persona describing the student   (osim_passive.txt)
    user      = the tutor's turns                    (the policy's output)
    assistant = the simulated student's own turns

Two mechanical consequences in `train_engagement_classroom.py`:
`_build_userlm_messages` must map teacher->`user` and student->`assistant`
(currently the reverse), and `ParallelvLLMInference(userlm_mode=...)` must be
**False** -- that flag exists only to prepend a BOS that UserLM's template
omits; OSIM is stock Qwen3 ChatML and takes the normal `llm.chat` path.

## DECIDED: the problem enters via OSIM's system prompt

Under UserLM the problem arrived through a synthetic student-first opening turn
(`_opening_turn`), which existed to keep UserLM in its student-first training
distribution. OSIM has no such constraint -- a GUIDED conversation is already
teacher-first, which is exactly OSIM's natural `user`-then-`assistant` order --
but that opening was also how the problem text got in front of the student.

**Design A (system_only): append the problem to the persona in OSIM's `system`
slot.** No synthetic turn, no harness-injected tutor text.

```python
system = f"{persona}\n\nThe problem you are working on:\n\n{problem}"
messages = [{"role": "system",    "content": system}]
          + [{"role": "user",     "content": <tutor turn>},      # policy output
             {"role": "assistant","content": <student turn>}, ...]
```

`persona` = `prompt_templates/personas/osim_passive.txt`. Nothing else is
prepended: the tutor speaks first and the student replies, which is both the
canonical GUIDED order and OSIM's native turn order.

### Why, and what was rejected

Measured in `utils/osim_problem_entry_probe.py` (n=40 problems x 2 real dialogue
turns against the actual Qwen2.5-7B teacher policy); results in
`logs/osim_problem_entry.json`. Grounding = share of numbers in the student's
turn that actually occur in the problem, measured on the discriminating subset
where the teacher restated <20% of the problem's numbers.

| condition | grounding | verdict |
|---|---:|---|
| A system_only | 0.44 | **adopted** |
| B tutor_only | 0.21 | rejected -- student invents its own problem |
| C tutor_explicit | 0.40 | rejected on design grounds (see below) |
| D synthetic_opening | 0.44 | rejected on design grounds (see below) |

- **B fails outright.** The teacher restates only 45% of the problem's numbers
  on average and fully restates just 26% of the time, so three quarters of
  dialogues would open with a student who does not know the task. Observed
  directly: problem states first term 1783 / last term 1993, teacher gives the
  formula without the numbers, and the B student answers
  *"I got 20 = 10 + 10d and 24 = 10 + 14d"* -- a fabricated problem. The same
  dialogue under A: *"I plug in 1783 for a1 and 1993 for an."*
- **C is disqualified for RL specifically.** It has the harness prefix the
  problem onto the tutor's *visible* turn, so the terminal learning judge would
  score a transcript containing text the policy never generated and
  credit-assign it to the policy. That is a credit-assignment bug, not a style
  preference.
- **D works empirically but fabricates a student turn.** Under UserLM that turn
  was view-only and never entered `conv.conversation`, so no judge saw it;
  preserving that invariant for OSIM is extra machinery for no measured gain
  over A.

Caveat on the metric: grounding counts a student's legitimately chosen test
value as ungrounded (*"I'll start with n=10"* scores 0.0 under every condition),
so 0.44 understates A. The B-vs-A contrast, not the absolute level, is what
carries the conclusion.

## Judge prompts: carry over, but two known soft spots

Both judges are transcript-only and model-agnostic, so they run against OSIM
unchanged. Two things to re-check rather than assume:

1. `ENGAGEMENT_SYSTEM_PROMPT`'s behavioral/affective **0 and 1 anchors** are
   written around a student who can leave ("ends the conversation", "signals
   wanting to stop"). OSIM never leaves, so those anchors may be unreachable
   and the realized scale compresses to 2-4. That interacts with
   `reward.engagement_weight`, which is NOT rescaled under
   `scale_rewards: False`.
2. Both prompts' ROLE CHECK / ROLE RULE describe assistant-drift in UserLM's
   idioms. OSIM drifts differently (and now occupies the `assistant` slot,
   which may make drift likelier), so the wording and the
   `looks_like_assistant` regex both need re-deriving from real OSIM drift
   samples.

## Eval-time Δ solve rate: use the NEUTRAL prompts, not the personas

`prompt_templates/student_initial_attempt_prompt.txt` and
`student_final_prompt.txt` (TutorRL verbatim, item 8 above) stay exactly as
they are -- just pointed at OSIM instead of Llama-3.1-8B. This removes the
probe/simulator mismatch, since the student being tutored and the student being
tested become the same model.

**Deliberately out of persona.** The passive persona declines the probe 15.6%
of the time, and refusals score as wrong, which would conflate "cannot solve"
with "would not attempt" (measured in `logs/osim_solve_probe_v2.json`).

## Files that are NOT part of training or eval

Written as measurement instruments only; do not wire these into a run:

- `prompt_templates/osim_initial_attempt_prompt.txt`
- `prompt_templates/osim_final_solution_prompt.txt`

These are the *in-persona* versions of the probe. Their only purpose was to
measure what happens when the solve request arrives as a tutor turn inside the
persona -- which is what produced the 15.6% refusal finding that justified
using the neutral prompts instead.

- `prompt_templates/personas/osim_engaged.txt` -- the second training arm
  (persona is fixed within a run and varied across runs; never varied within a
  GRPO group, or it becomes an unobserved confounder inflating within-group
  variance with zero tutor signal). Not used until the passive run is done.
- `prompt_templates/personas/simple_student.txt` -- TutorRL's always-engaged
  baseline student. Unrelated to OSIM.
