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
