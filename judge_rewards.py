"""Judge prompts + parsing for engagement-conditioned RL training rewards.

Two judge calls (both gpt-oss-20b, transcript-only):
  1. PER-TURN engagement: the validated 3-dim rubric from engagement_eval
     (behavioral/affective/cognitive).
  2. PER-TURN learning outcome: 3-dim rubric (solution progress /
     understanding / misconception repair), scored on the same
     dialogue-so-far + target-turn view as the engagement judge. Leakage is
     penalised inside solution_progress (a tutor-stated answer scores 0) and
     flagged separately via tutor_leaked.

Scalars are normalized to [0,1]; weighting happens in the trainer.
"""
import json
import re

# ---------------------------------------------------------------------------
# Per-turn prompt: engagement (validated rubric, verbatim)
# ---------------------------------------------------------------------------

ENGAGEMENT_SYSTEM_PROMPT = """\
You are an expert in the learning sciences annotating STUDENT engagement in a \
one-on-one tutoring dialogue. You will see a dialogue transcript up to and \
including one STUDENT turn (the "target turn"). Rate the engagement expressed by \
the student IN THE TARGET TURN, given the conversation so far.

Engagement is a multidimensional construct. Score three dimensions independently. \
Judge ONLY the student. Ignore the tutor's quality. Productive confusion counts as \
ENGAGED, not disengaged; only checking-out counts as disengaged.

=== BEHAVIORAL (participation & persistence) ===
4  Actively drives the interaction forward: attempts the step, asks to keep going, volunteers work unprompted, shows increased interest in the topic, ask for related concepts.
3  Participates substantively in response to each tutor prompt; stays on task.
2  Answers technical tutor questions, albeit simply.
1  Passive compliance: minimal responses ("ok", "sure", "next"), demonstrate some engagement with the topic.
0  Disengaged/absent: demands the answer instead of working, signals wanting to stop,
   terse dead-end replies, refuses to continue, off-task, or ends the conversation.

=== AFFECTIVE (interest vs. boredom/frustration) ===
4  Positive interest/curiosity ("oh interesting", "so is it because...?").
3  Willing, neutral-positive tone.
2  Flat/neutral affect.
1  Negative but still present: boredom, impatience, or frustration WHILE still trying. (Frustration + effort belongs here, not 0.)
0  Strong disengaged affect: annoyed/dismissive, "this is pointless", checked out.

=== COGNITIVE (depth of processing) ===
4  Interactive: integrates the tutor's hint into the student's own reasoning; co-constructs, builds a genuine back-and-forth on the idea.
3  Constructive: generates NEW reasoning beyond what was given -- a justification, an inference, a genuine conceptual question, an unprompted step.
2  Active: manipulates given material -- plugs in numbers, attempts the tutor's question, picks from options.
1  Passive: only receives/acknowledges ("got it", "I see") with no processing.
0  None: ignores the tutor's content, or just re-demands the answer.

=== ROLE CHECK ===
The "student" is a simulated user and occasionally malfunctions by slipping
into an AI-assistant persona: offering help, tutoring the other party, walking
through numbered solution steps addressed at the tutor, or closing with
service phrases ("hope this helps", "feel free to ask"). If the TARGET TURN
reads as an assistant helping someone rather than a student seeking help or
working on their own problem, set drift=true. A student legitimately doing the
math (even a full solution attempt in their own voice) is NOT drift.

For EACH dimension, give a one-clause rationale, then the score. If the target
turn is empty or pure silence, all three scores are 0.

Return ONLY this JSON object and nothing else:
{"behavioral": {"rationale": "...", "score": N},
 "affective": {"rationale": "...", "score": N},
 "cognitive": {"rationale": "...", "score": N},
 "role_drift": {"rationale": "...", "drift": true_or_false}}"""

ENGAGEMENT_DIMS = ("behavioral", "affective", "cognitive")

# ---------------------------------------------------------------------------
# Turn-level learning outcome
# ---------------------------------------------------------------------------

LEARNING_SYSTEM_PROMPT = """\
You are an expert in the learning sciences annotating STUDENT engagement in a \
one-on-one tutoring dialogue. You will see a dialogue transcript up to and \
including one STUDENT turn (the "target turn"). You will see the
problem, its ground-truth answer, and the full transcript. Assess how much the
STUDENT demonstrably learned in this turn.

Additionally perform a LEAKAGE CHECK on the TUTOR: set leaked=true if the
tutor gave away the solution rather than guiding the student to it -- e.g.
stated the final answer before the student produced it, revealed all key
solution steps in a single message, or walked through the solution without
meaningful student involvement. Guiding interactively, confirming the
student's own correct results, correcting mistakes, and clarifying concepts
gradually are NOT leakage.

Score four dimensions independently:

=== SOLUTION PROGRESS (student-produced progress on the problem) ===
4  Student produces a complete correct solution after integrating previous steps.
3  Student correctly executes the key step(s) they initially did wrong.
2  Partial progress: some correct student-produced steps; gaps remain.
1  Attempts exist but are wrong, or student merely repeats what the tutor dictates.
0  No student-produced progress (including: correct answer present in the
   transcript but tutor-stated).

=== UNDERSTANDING DEMONSTRATED (depth, not correctness alone) ===
4  Generalizes: articulates the underlying principle or when the method applies
   beyond this problem.
3  Explains WHY a step works; connects steps.
2  Applies a shown step correctly to new numbers.
1  Rote: repeats the tutor's words or steps without transformation.
0  No comprehension evidence, or persistent misunderstanding.

=== MISCONCEPTION REPAIR ===
First identify the student's initial error(s), if any.
4  Student identifies their own earlier error and articulates the fix.
3  Initial error is corrected IN THE STUDENT'S later work.
2  Error acknowledged after the tutor flags it; sensible fix attempted.
1  Error persists in modified form.
0  Initial errors unaddressed or repeated.

Return ONLY this JSON object and nothing else:
{"solution_progress": {"rationale": "...", "score": N},
 "understanding": {"rationale": "...", "score": N},
 "misconception_repair": {"rationale": "...", "score": N},
 "tutor_leaked": {"rationale": "...", "leaked": true_or_false}}"""

LEARNING_DIMS = (
    "solution_progress",
    "understanding",
    "misconception_repair",
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


def build_learning_user_prompt(problem, answer, context_turns, target_text):
    """PER-TURN user message for the learning judge: same shape as the
    engagement prompt (dialogue so far + one target student turn), plus the
    problem and ground-truth answer the learning rubric needs.

    Was previously whole-dialogue and scored once at the end. Per-turn so it can
    feed per-turn credit assignment, and because the terminal framing asked the
    judge to score change "relative to the student's first attempt" -- an
    artifact that does not exist at training time (no pre-dialogue probe is
    computed, and GUIDED dialogues are teacher-first, so even the student's
    first turn is already a response to tutoring)."""
    ctx = context_turns
    dropped = len(ctx) - MAX_CTX_TURNS
    if dropped > 0:
        ctx = ctx[-MAX_CTX_TURNS:]
    lines = [f"[PROBLEM]\n{problem}", f"\n[GROUND-TRUTH ANSWER]\n{answer}",
             "\n[DIALOGUE SO FAR]"]
    if dropped > 0:
        lines.append(f"(... {dropped} earlier turns omitted ...)")
    for m in ctx:
        content = m["content"].strip() or "(says nothing and leaves)"
        lines.append(f"{_ROLE_NAME[m['role']]}: {_trunc(content)}")
    lines.append("\n[TARGET STUDENT TURN -- rate this]")
    lines.append(f"Student: {_trunc(target_text)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing (gpt-oss may emit analysis before the final JSON message)
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.S)


def _extract_scores(text, dims):
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
                if not 0 <= s <= 4:
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
    return _extract_scores(text, LEARNING_DIMS)


# Fallbacks when the judge fails to produce parsable JSON twice. Midpoint for
# engagement (no fake advantage in either direction); conservative low for
# learning evidence / outcome.
DEFAULT_ENGAGEMENT_SCORES = {
    "behavioral": 2, "affective": 2, "cognitive": 2, "role_drift": False,
}
DEFAULT_LEARNING_SCORES = {
    "solution_progress": 1, "understanding": 1, "misconception_repair": 1,
    "tutor_leaked": False,
}

ZERO_ENGAGEMENT_SCORES = {**{k: 0 for k in ENGAGEMENT_DIMS}, "role_drift": False}
ZERO_LEARNING_SCORES = {**{k: 0 for k in LEARNING_DIMS}, "tutor_leaked": False}


# ---------------------------------------------------------------------------
# Scalar rewards in [0,1]
# ---------------------------------------------------------------------------

def engagement_scalar(turn_scores, max_student_turns=None):
    """Summed per-turn engagement (behavioral+affective+cognitive)/12,
    normalized by the maximum number of student turns the episode could have
    had rather than by the number it actually had.

    Why not a plain mean: a mean is length-blind, so a dialogue where the
    student leaves after one turn scores the same as one where they stayed for
    seven at equal per-turn quality. Only *silent* exits were penalized (via
    ZERO_ENGAGEMENT_SCORES); a student who signs off with real text
    ("thanks, got it") kept full credit. Normalizing by the episode's capacity
    makes turns the student never took forfeit their credit, which is the
    closest a judge-only reward gets to the survival factor in the
    multiplicative objective.

    Turns the judge flagged as assistant role-drift are excluded from BOTH the
    numerator and the denominator, preserving the existing decision that drift
    is simulator noise -- neither rewarded nor penalized. Were they left in the
    denominator, a drifted turn would silently cost reward.

    max_student_turns=None keeps the original mean-over-actual-turns behaviour
    (used by callers that have no generation config to hand).
    """
    kept = [t for t in turn_scores if not t.get("role_drift", False)]
    if not kept:
        return 0.0
    vals = [
        (t["behavioral"] + t["affective"] + t["cognitive"]) / 12.0
        for t in kept
    ]
    if max_student_turns is None:
        return sum(vals) / len(vals)
    n_drifted = len(turn_scores) - len(kept)
    denom = max(1, max_student_turns - n_drifted)
    # Clamp: a conversation cannot exceed its own capacity, but token-budget
    # truncation and odd max_turns values make this worth asserting cheaply.
    return min(1.0, sum(vals) / denom)


def learning_scalar(scores):
    """Mean of the 3 learning dims / 4 -> [0,1].

    misconception_repair no longer has an N/A (-1) escape: it was returned in
    93% of dialogues (the student must first make an error for there to be one
    to repair), so the dimension almost never contributed. It is now scored on
    the same 0-4 scale as the others.
    `independence` is gone: it scored 0 in 99% of dialogues because it is a
    trajectory property ("scaffolding fades") that 2-turn dialogues cannot show.
    """
    if scores is None:
        return 0.0
    return sum(scores[d] for d in LEARNING_DIMS) / (4.0 * len(LEARNING_DIMS))


# ---------------------------------------------------------------------------
# Reusable scoring over completed dialogues (shared by training rewards and
# eval metrics, so the eval numbers use the SAME rubrics as the reward).
# ---------------------------------------------------------------------------

def judge_dialogues(run_batch, dialogues, problems, answers):
    """Score completed dialogues with both rubrics, BOTH per student turn.

    run_batch: callable(list_of_chat_message_lists) -> list of raw text outputs.
    dialogues: list of [{'role': 'teacher'|'student', 'content': str}] with
        thinking already stripped.

    Returns (turn_scores, learning_turn_scores), each a per-dialogue list of
    per-student-turn score dicts. Aggregate with per_turn_rewards() or the
    scalars; do not report individual turns as a metric.
    """
    def _batch(messages, extractor, defaults):
        if not messages:
            return []
        parsed = [extractor(t) for t in run_batch(messages)]
        fail = [i for i, p in enumerate(parsed) if p is None]
        if fail:
            retry = run_batch([messages[i] for i in fail])
            for i, t in zip(fail, retry):
                parsed[i] = extractor(t)
        return [p if p is not None else dict(defaults) for p in parsed]

    eng_msgs, lrn_msgs, slots = [], [], []
    turn_scores = [[] for _ in dialogues]
    learning_turn_scores = [[] for _ in dialogues]
    for di, turns in enumerate(dialogues):
        for i, m in enumerate(turns):
            if m["role"] != "student":
                continue
            idx = len(turn_scores[di])
            if not m["content"].strip():
                turn_scores[di].append(dict(ZERO_ENGAGEMENT_SCORES))
                learning_turn_scores[di].append(dict(ZERO_LEARNING_SCORES))
                continue
            eng_msgs.append([
                {"role": "system", "content": ENGAGEMENT_SYSTEM_PROMPT},
                {"role": "user",
                 "content": build_engagement_user_prompt(turns[:i], m["content"])},
            ])
            lrn_msgs.append([
                {"role": "system", "content": LEARNING_SYSTEM_PROMPT},
                {"role": "user", "content": build_learning_user_prompt(
                    problems[di], answers[di], turns[:i], m["content"])},
            ])
            slots.append((di, idx))
            turn_scores[di].append(None)
            learning_turn_scores[di].append(None)

    for (di, idx), sc in zip(slots, _batch(eng_msgs, extract_engagement_scores,
                                           DEFAULT_ENGAGEMENT_SCORES)):
        turn_scores[di][idx] = sc
    for (di, idx), sc in zip(slots, _batch(lrn_msgs, extract_learning_scores,
                                           DEFAULT_LEARNING_SCORES)):
        learning_turn_scores[di][idx] = sc
    return turn_scores, learning_turn_scores


# ---------------------------------------------------------------------------
# Per-turn rewards (dense credit assignment)
# ---------------------------------------------------------------------------

def per_turn_rewards(turn_scores, learning_turn_scores=None,
                     learning_weight=1.0, engagement_weight=0.5):
    """One reward per STUDENT turn, for per-turn credit assignment.

        r_t = w_L * ((solution_progress+understanding+misconception_repair)_t / 12)
            + w_E * ((behavioral+affective+cognitive)_t / 12)

    Both rubrics are now scored PER TURN, so both components are 3 dims x 0-4
    and divide by 12 to land in [0,1]; w_L and w_E therefore mean the same
    thing as in the trajectory-level path.

    Teacher turn t is credited with the STUDENT turn it elicited. turn_scores
    and learning_turn_scores are positionally aligned with the student turns,
    and the trainer maps each teacher turn to the same index.

    No terminal term: the learning rubric is per-turn now, so there is no
    separate dialogue-level score to fold in. No length term: it was fighting
    the learning signal (leaked dialogues run longer and collected more length
    bonus, which outweighed the +0.019 evidence advantage of not leaking), and
    turn-position credit is handled by the trainer's baseline instead.

    Drift-flagged turns get 0.0 so they neither reward nor penalise.
    """
    lts = learning_turn_scores or []
    out = []
    for i, t in enumerate(turn_scores):
        if t.get("role_drift", False):
            out.append(0.0)
            continue
        eng = sum(t[d] for d in ENGAGEMENT_DIMS) / 12.0
        learn = learning_scalar(lts[i]) if i < len(lts) else 0.0
        out.append(learning_weight * learn + engagement_weight * eng)
    return out
