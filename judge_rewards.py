"""Judge prompts + parsing for engagement-conditioned RL training rewards.

Two judge calls (both gpt-oss-20b, transcript-only):
  1. PER-TURN engagement: the validated 3-dim rubric from engagement_eval
     (behavioral/affective/cognitive, 0-4 anchored) plus a 4th per-turn
     LEARNING EVIDENCE dimension (new correct student-produced reasoning).
  2. TERMINAL learning outcome: 4-dim dialogue-level rubric
     (solution progress / understanding / misconception repair / independence)
     with an evidence rule (tutor-stated answers score 0 -> leakage gets no
     learning credit) and a delta rule (score change, not ability).

Scalars are normalized to [0,1]; weighting happens in the trainer.
"""
import json
import re

# ---------------------------------------------------------------------------
# Per-turn prompt: engagement (validated rubric, verbatim) + learning evidence
# ---------------------------------------------------------------------------

ENGAGEMENT_SYSTEM_PROMPT = """\
You are an expert in the learning sciences annotating STUDENT engagement in a \
one-on-one tutoring dialogue. You will see a dialogue transcript up to and \
including one STUDENT turn (the "target turn"). Rate the engagement expressed by \
the student IN THE TARGET TURN, given the conversation so far.

Engagement is a multidimensional construct. Score three dimensions independently. \
Judge ONLY the student. Ignore the tutor's quality. Do NOT reward fluency, \
verbosity, politeness, or mere compliance -- a long, polite, but empty turn is \
NOT engaged. Productive confusion ("wait, why does that work?") counts as \
ENGAGED, not disengaged; only checking-out counts as disengaged.

=== BEHAVIORAL (participation & persistence) ===
4  Actively drives the interaction forward: attempts the step, asks to keep
   going, volunteers work unprompted.
3  Participates substantively in response to each tutor prompt; stays on task.
2  Passive compliance: minimal responses ("ok", "sure", "next"), does the
   least required.
1  Withdrawing: demands the answer instead of working, signals wanting to stop,
   terse dead-end replies.
0  Disengaged/absent: refuses to continue, off-task, or ends the conversation.

=== AFFECTIVE (interest vs. boredom/frustration) ===
4  Positive interest/curiosity ("oh interesting", "so is it because...?").
3  Willing, neutral-positive tone.
2  Flat/neutral affect.
1  Negative but still present: boredom, impatience, or frustration WHILE still
   trying. (Frustration + effort belongs here, not 0.)
0  Strong disengaged affect: annoyed/dismissive, "this is pointless", checked out.

=== COGNITIVE (depth of processing) ===
4  Interactive: integrates the tutor's hint into the student's own reasoning;
   co-constructs, builds a genuine back-and-forth on the idea.
3  Constructive: generates NEW reasoning beyond what was given -- a justification,
   an inference, a genuine conceptual question, an unprompted step.
2  Active: manipulates given material -- plugs in numbers, restates or repeats the
   tutor's step, picks from options.
1  Passive: only receives/acknowledges ("got it", "I see") with no processing.
0  None: ignores the tutor's content, or just re-demands the answer.

=== LEARNING EVIDENCE (new understanding shown in THIS turn) ===
Score whether the target turn contains NEW, CORRECT, student-produced reasoning
that was not present earlier in the dialogue. Credit only what the STUDENT
produces. Content merely restated from the tutor scores at most 1. If the tutor
already stated the answer or the full step, the student repeating it scores 0-1
regardless of correctness.
4  New correct reasoning: an unprompted correct step, a correct self-explanation
   of WHY something works, or spotting and fixing their own earlier error.
3  Correct completion of a step the tutor set up, with some student
   transformation (own words, own algebra), beyond copying.
2  Correct mechanical application: plugs numbers into a method just shown;
   right answer, no visible understanding beyond execution.
1  Attempted but incorrect new work, or verbatim echo of the tutor.
0  No new content: acknowledgment, question, off-task, or repeats an earlier
   error.
For learning_evidence you MUST include a short verbatim quote from the target
turn as evidence; if no quote supports a score above 0, the score is 0.

=== ROLE CHECK ===
The "student" is a simulated user and occasionally malfunctions by slipping
into an AI-assistant persona: offering help, tutoring the other party, walking
through numbered solution steps addressed at the tutor, or closing with
service phrases ("hope this helps", "feel free to ask"). If the TARGET TURN
reads as an assistant helping someone rather than a student seeking help or
working on their own problem, set drift=true. A student legitimately doing the
math (even a full solution attempt in their own voice) is NOT drift.

For EACH dimension, give a one-clause rationale, then the score. If the target
turn is empty or pure silence, all four scores are 0.

Return ONLY this JSON object and nothing else:
{"behavioral": {"rationale": "...", "score": N},
 "affective": {"rationale": "...", "score": N},
 "cognitive": {"rationale": "...", "score": N},
 "learning_evidence": {"quote": "...", "rationale": "...", "score": N},
 "role_drift": {"rationale": "...", "drift": true_or_false}}"""

ENGAGEMENT_DIMS = ("behavioral", "affective", "cognitive", "learning_evidence")

# ---------------------------------------------------------------------------
# Terminal prompt: dialogue-level learning outcome
# ---------------------------------------------------------------------------

LEARNING_SYSTEM_PROMPT = """\
You are an expert in the learning sciences assessing LEARNING OUTCOMES from a
completed one-on-one tutoring dialogue about a problem. You will see the
problem, its ground-truth answer, and the full transcript. Assess how much the
STUDENT demonstrably learned DURING the dialogue.

Two strict rules:
(1) EVIDENCE RULE: credit only what the student demonstrably produced or
    articulated. A correct answer stated by the TUTOR is worth NOTHING, even if
    the student says "got it" or repeats it. If you cannot quote student text
    as evidence, the score is 0.
(2) DELTA RULE: score learning (change during the dialogue), not ability. A
    student who already solves the problem correctly at the start has shown
    competence, not learning; score what changed relative to their first
    attempt.
(3) ROLE RULE: the student is simulated and may malfunction into an
    AI-assistant persona (tutoring the tutor, numbered walkthroughs addressed
    at the other party, "hope this helps"). Such turns are simulator noise:
    credit NO learning from them on any dimension.

Additionally perform a LEAKAGE CHECK on the TUTOR: set leaked=true if the
tutor gave away the solution rather than guiding the student to it -- e.g.
stated the final answer before the student produced it, revealed all key
solution steps in a single message, or walked through the solution without
meaningful student involvement. Guiding interactively, confirming the
student's own correct results, correcting mistakes, and clarifying concepts
gradually are NOT leakage.

Score four dimensions independently:

=== SOLUTION PROGRESS (student-produced progress on the problem) ===
4  Student produces a complete correct solution largely by their own work.
3  Student correctly executes the key step(s) they initially could not do.
2  Partial progress: some correct student-produced steps; gaps remain.
1  Attempts exist but steps are wrong, or student merely transcribes what the
   tutor dictates.
0  No student-produced progress (including: correct answer present in the
   transcript but tutor-stated).

=== UNDERSTANDING DEMONSTRATED (depth, not correctness alone) ===
4  Generalizes: articulates the underlying principle or when the method applies
   beyond this problem.
3  Explains WHY a step works in their own words; connects steps.
2  Applies a shown step correctly to new numbers but never justifies it.
1  Rote: repeats the tutor's words or steps without transformation.
0  No comprehension evidence, or persistent misunderstanding.

=== MISCONCEPTION REPAIR ===
First identify the student's initial error(s), if any.
4  Student identifies their own earlier error and articulates the fix.
3  Initial error is corrected IN THE STUDENT'S later work.
2  Error acknowledged after the tutor flags it; weakly re-demonstrated.
1  Error persists in modified form, or only the tutor corrects it.
0  Initial errors unaddressed or repeated.
If the student made no initial error, output score -1 (not applicable).

=== INDEPENDENCE TRAJECTORY (scaffolding fading) ===
4  Tutor support fades; student drives the final steps unprompted.
3  Student needs progressively smaller hints for comparable steps.
2  Constant level of tutor support throughout.
1  Student needs increasing support, or offloads more over time.
0  Tutor does essentially all the work end to end.

For EACH dimension give one short student quote as evidence (or "" if none), a
one-clause rationale, then the score.

Return ONLY this JSON object and nothing else:
{"solution_progress": {"quote": "...", "rationale": "...", "score": N},
 "understanding": {"quote": "...", "rationale": "...", "score": N},
 "misconception_repair": {"quote": "...", "rationale": "...", "score": N},
 "independence": {"quote": "...", "rationale": "...", "score": N},
 "tutor_leaked": {"rationale": "...", "leaked": true_or_false}}"""

LEARNING_DIMS = (
    "solution_progress",
    "understanding",
    "misconception_repair",
    "independence",
)

# ---------------------------------------------------------------------------
# User-message builders (same formatting conventions as engagement_eval)
# ---------------------------------------------------------------------------

MAX_CTX_TURNS = 10
MAX_TURN_CHARS = 1200

_ROLE_NAME = {"teacher": "Tutor", "student": "Student"}


def _trunc(t, limit=MAX_TURN_CHARS):
    return t if len(t) <= limit else t[:limit] + " [...truncated]"


def build_engagement_user_prompt(context_turns, target_text):
    """context_turns: [{'role': 'teacher'|'student', 'content': str}, ...]
    (thinking already hidden); target_text: the student turn to rate."""
    ctx = context_turns
    dropped = len(ctx) - MAX_CTX_TURNS
    if dropped > 0:
        ctx = ctx[-MAX_CTX_TURNS:]
    lines = ["[DIALOGUE SO FAR]"]
    if dropped > 0:
        lines.append(f"(... {dropped} earlier turns omitted ...)")
    for m in ctx:
        lines.append(f"{_ROLE_NAME[m['role']]}: {_trunc(m['content'])}")
    lines.append("\n[TARGET STUDENT TURN -- rate this]")
    lines.append(f"Student: {_trunc(target_text)}")
    return "\n".join(lines)


def build_learning_user_prompt(problem, answer, turns):
    """Full-dialogue user message for the terminal learning-outcome judge."""
    lines = [f"[PROBLEM]\n{problem}", f"\n[GROUND-TRUTH ANSWER]\n{answer}",
             "\n[FULL TRANSCRIPT]"]
    for m in turns:
        content = m["content"].strip() or "(says nothing and leaves)"
        lines.append(f"{_ROLE_NAME[m['role']]}: {_trunc(content, 2000)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing (gpt-oss may emit analysis before the final JSON message)
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.S)


def _extract_scores(text, dims, allow_na=()):
    for cand in reversed(_JSON_RE.findall(text)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not all(k in obj for k in dims):
            continue
        try:
            out = {}
            for k in dims:
                s = int(obj[k]["score"])
                lo = -1 if k in allow_na else 0
                if not lo <= s <= 4:
                    raise ValueError
                out[k] = s
            # Optional judge-side drift flag (backstop to the rollout-time
            # regex detector); absent/malformed -> False.
            try:
                out["role_drift"] = bool(obj["role_drift"]["drift"])
            except (KeyError, TypeError):
                out["role_drift"] = False
            try:
                out["tutor_leaked"] = bool(obj["tutor_leaked"]["leaked"])
            except (KeyError, TypeError):
                out["tutor_leaked"] = False
            return out
        except (KeyError, TypeError, ValueError):
            continue
    return None


def extract_engagement_scores(text):
    return _extract_scores(text, ENGAGEMENT_DIMS)


def extract_learning_scores(text):
    return _extract_scores(text, LEARNING_DIMS, allow_na=("misconception_repair",))


# Fallbacks when the judge fails to produce parsable JSON twice. Midpoint for
# engagement (no fake advantage in either direction); conservative low for
# learning evidence / outcome.
DEFAULT_ENGAGEMENT_SCORES = {
    "behavioral": 2, "affective": 2, "cognitive": 2, "learning_evidence": 1,
    "role_drift": False,
}
DEFAULT_LEARNING_SCORES = {
    "solution_progress": 1, "understanding": 1,
    "misconception_repair": -1, "independence": 2,
    "tutor_leaked": False,
}

ZERO_ENGAGEMENT_SCORES = {**{k: 0 for k in ENGAGEMENT_DIMS}, "role_drift": False}


# ---------------------------------------------------------------------------
# Scalar rewards in [0,1]
# ---------------------------------------------------------------------------

def engagement_scalar(turn_scores):
    """Mean over student turns of (behavioral+affective+cognitive)/12.
    Turns the judge flagged as assistant role-drift are excluded (simulator
    noise, neither rewarded nor penalized). Empty/all-drifted -> 0."""
    kept = [t for t in turn_scores if not t.get("role_drift", False)]
    if not kept:
        return 0.0
    vals = [
        (t["behavioral"] + t["affective"] + t["cognitive"]) / 12.0
        for t in kept
    ]
    return sum(vals) / len(vals)


def learning_scalar(scores):
    """Mean of applicable dims / 4; misconception_repair==-1 is dropped."""
    if scores is None:
        return 0.0
    dims = [scores["solution_progress"], scores["understanding"],
            scores["independence"]]
    if scores["misconception_repair"] >= 0:
        dims.append(scores["misconception_repair"])
    return sum(dims) / (4.0 * len(dims))
