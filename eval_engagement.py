"""Eval driver for the engagement-conditioned baseline (TutorRL x UserLM).

Same metrics as eval.py (pre/post solve rate -> Delta Solve Rate, Leaked Solution,
pedagogy) plus engagement metrics: disengagement rate and turns-to-disengagement.
Run with env_eval.sh sourced and PYTHONUNBUFFERED=1.
"""
import os
import hydra
import warnings
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from dotenv import load_dotenv

from engagement_classroom import EngagementClassroom
from osim_classroom import OsimEngagementClassroom
from src.classroom import JudgeDecision
from judge_rewards import (
    engagement_scalar,
    judge_dialogues,
    learning_scalar,
)
from utils.data import load_datasets
from utils.transfer_test import load_sibling_pool, select_siblings
from config.eval import EvalConfig
from config.train_rl_model import StudentModelConfig
from src.utils.utils import init_logger

load_dotenv()
logger = init_logger()
warnings.filterwarnings("ignore")


@dataclass
class EngagementEvalConfig(EvalConfig):
    # UserLM-8b drives the in-dialogue student turns (engagement simulator).
    engagement_model: StudentModelConfig = field(
        default_factory=lambda: StudentModelConfig(
            model_name_or_path="microsoft/UserLM-8b"
        )
    )
    # Which simulator drives the in-dialogue student: "userlm" or "osim".
    # Only the classroom changes -- teacher, knowledge probe, judge, dataset
    # and metrics are identical, so the two are a controlled pair.
    simulator: str = "userlm"
    # OSIM only: persona in the system slot (see PROMPTS.md, design A).
    persona_path: str = "prompt_templates/personas/osim_passive.txt"


cs = ConfigStore.instance()
cs.store(name="config", node=EngagementEvalConfig)


def _macro_mean(conversations, n_problems, n_samples, fn):
    vals = []
    for i in range(n_problems):
        cur = [fn(conversations[i * n_samples + j]) for j in range(n_samples)]
        cur = [c for c in cur if c is not None]
        if cur:
            vals.append(sum(cur) / len(cur))
    return sum(vals) / len(vals) if vals else float("nan")


def _macro_mean_list(values, n_problems, n_samples):
    """Same nested-averaging shape as _macro_mean, but over a flat list of
    scalars already aligned to the sample_problems/sample_answers ordering,
    rather than a per-conversation callback."""
    vals = []
    for i in range(n_problems):
        cur = [v for v in values[i * n_samples : (i + 1) * n_samples] if v is not None]
        if cur:
            vals.append(sum(cur) / len(cur))
    return sum(vals) / len(vals) if vals else float("nan")


@hydra.main(config_path="config/eval", version_base=None)
def main(cfg):
    cfg = OmegaConf.merge(OmegaConf.structured(EngagementEvalConfig), cfg)

    if cfg.simulator == "osim":
        logger.info(f"Engagement simulator: OSIM (persona {cfg.persona_path})")
        classroom = OsimEngagementClassroom(
            cfg.student_model,
            cfg.teacher_model,
            cfg.judge_model,
            cfg.reward_model,
            cfg.generation,
            None,
            cfg.engagement_model,
            persona_path=cfg.persona_path,
        )
    elif cfg.simulator == "userlm":
        classroom = EngagementClassroom(
            cfg.student_model,
            cfg.teacher_model,
            cfg.judge_model,
            cfg.reward_model,
            cfg.generation,
            None,
            cfg.engagement_model,
        )
    else:
        raise ValueError(
            f"Unknown simulator {cfg.simulator!r}; expected 'userlm' or 'osim'"
        )

    _, eval_data = load_datasets(cfg.dataset, cfg.seed)
    # Capture domain/difficulty metadata before eval_data gets sliced down to
    # plain problem/answer lists below -- needed for sibling selection.
    domains, solve_rates = eval_data["domain"], eval_data["llama8b_solve_rate"]
    problems, answers = eval_data["problem"], eval_data["answer"]
    n_problems = len(problems)
    n_samples = cfg.num_samples_per_problem

    sample_problems, sample_answers = [], []
    for i in range(n_problems):
        sample_problems.extend([problems[i]] * n_samples)
        sample_answers.extend([answers[i]] * n_samples)

    logger.info("Selecting transfer-test sibling problems...")
    dataset_name = cfg.dataset.eval_datasets[0].name_or_path
    sibling_pool = load_sibling_pool(dataset_name, exclude_problems=set(problems))
    sibling_problems, sibling_answers, sibling_fallback = select_siblings(
        problems, domains, solve_rates, sibling_pool, cfg.seed
    )
    n_domain_fallback = sum(1 for t in sibling_fallback if t.startswith("no-domain"))
    logger.info(
        f"Sibling selection: {n_domain_fallback}/{n_problems} problems had no "
        "same-domain match within tolerance and fell back to difficulty-only matching."
    )
    sample_sibling_problems, sample_sibling_answers = [], []
    for i in range(n_problems):
        sample_sibling_problems.extend([sibling_problems[i]] * n_samples)
        sample_sibling_answers.extend([sibling_answers[i]] * n_samples)

    logger.info("Sampling conversations (UserLM student)...")
    conversations = classroom.sample_conversations(
        sample_problems, sample_answers, compute_initial_attempt=True
    )

    logger.info("Running transfer test (treatment: sibling solved in-context)...")
    classroom.run_transfer_test(
        conversations, sample_sibling_problems, sample_sibling_answers
    )

    logger.info("Running transfer test (control: sibling solved cold)...")
    control_rewards = classroom.compute_cold_solve_rewards(
        sample_sibling_problems, sample_sibling_answers
    )

    logger.info("Computing metrics...")
    initial = _macro_mean(
        conversations, n_problems, n_samples, lambda c: c.get_initial_rm_reward()
    )
    end = _macro_mean(
        conversations, n_problems, n_samples, lambda c: c.get_end_rm_reward()
    )
    delta = end - initial

    def leaked(c):
        d = [x.decision for x in c.judge_decisions.get("does_not_leak_answer", [])]
        return d.count(JudgeDecision.REJECT) / len(d) if d else None

    leaked_mean = _macro_mean(conversations, n_problems, n_samples, leaked)

    # Transfer test: sibling problem solved in-context after the dialogue
    # (treatment) vs. solved cold with no dialogue at all (control). Isolates
    # genuine transferable learning from same-problem leak/memorization
    # inflation, since the sibling has different numbers/surface form.
    transfer_treatment = _macro_mean(
        conversations, n_problems, n_samples, lambda c: c.get_transfer_rm_reward()
    )
    transfer_control = _macro_mean_list(control_rewards, n_problems, n_samples)
    transfer_delta = transfer_treatment - transfer_control

    # Engagement metrics
    disengaged_flags = [getattr(c, "disengaged", False) for c in conversations]
    disengage_rate = sum(disengaged_flags) / len(disengaged_flags)
    turns_all = [len(c.conversation) for c in conversations]
    avg_turns = sum(turns_all) / len(turns_all)
    disengage_turns = [
        getattr(c, "disengage_turn", None)
        for c in conversations
        if getattr(c, "disengaged", False)
    ]
    avg_disengage_turn = (
        sum(disengage_turns) / len(disengage_turns) if disengage_turns else float("nan")
    )

    # ---- LM-judged dialogue-level engagement + learning ----
    # Same rubrics as the training reward (judge_rewards.py), so eval numbers
    # are directly interpretable against what GRPO optimised. Reported per
    # DIALOGUE: the engagement rubric rates individual student turns, but only
    # its aggregate is a metric. Normalised by max student turns (max_turns//2)
    # rather than by turns taken, matching engagement_scalar's training use --
    # otherwise a one-turn dialogue and a full-length one score identically.
    # NOTE: this judge is the eval judge (see config), not the training judge,
    # so absolute values are not comparable across the two pipelines.
    logger.info("Scoring dialogues with the training rubrics (engagement + learning)...")
    max_student_turns = max(1, cfg.generation.max_turns // 2)

    def _judge_run_batch(messages):
        responses = classroom.judge_model.run_batch(
            messages, classroom.sampling_params_judge
        )
        return [r.outputs[0].text for r in responses]

    dialogues = [c._get_hidden_conversation() for c in conversations]
    turn_scores, learning_scores = judge_dialogues(
        _judge_run_batch, dialogues, sample_problems, sample_answers
    )
    eng_vals = [engagement_scalar(ts, max_student_turns) for ts in turn_scores]
    learn_vals = [learning_scalar(ls) for ls in learning_scores]
    judged_engagement = _macro_mean_list(eng_vals, n_problems, n_samples)
    judged_learning = _macro_mean_list(learn_vals, n_problems, n_samples)

    def _dim_mean(scores_list, dim, skip_negative=False):
        vals = [s[dim] for s in scores_list
                if not (skip_negative and s[dim] < 0)]
        return sum(vals) / len(vals) if vals else float("nan")

    flat_turns = [t for ts in turn_scores for t in ts if not t.get("role_drift")]

    print("\n================ ENGAGEMENT BASELINE RESULTS ================", flush=True)
    print(f"Problems x samples         : {n_problems} x {n_samples}", flush=True)
    print(f"Pre-dialog solve rate      : {initial:.4f}", flush=True)
    print(f"Post-dialog solve rate     : {end:.4f}", flush=True)
    print(f"Delta Solve Rate           : {delta:+.4f}", flush=True)
    print(f"Leaked Solution rate       : {leaked_mean:.4f}", flush=True)
    n_silence = sum(
        1 for c in conversations if getattr(c, "disengage_reason", None) == "silence"
    )
    n_explicit = sum(
        1
        for c in conversations
        if getattr(c, "disengage_reason", None) == "endconversation"
    )
    print(f"Disengagement rate         : {disengage_rate:.4f}", flush=True)
    print(
        f"  - via <endconversation>  : {n_explicit}/{len(conversations)}", flush=True
    )
    print(f"  - via silence (blank)    : {n_silence}/{len(conversations)}", flush=True)
    print(f"Avg dialog length (msgs)   : {avg_turns:.2f}", flush=True)
    print(f"Avg turns-to-disengagement : {avg_disengage_turn:.2f}", flush=True)

    print("--- LM-judged (same rubrics as training reward) ---", flush=True)
    print(f"Engagement (dialogue, /max turns)           : {judged_engagement:.4f}", flush=True)
    print(f"Learning   (dialogue, 4-dim rubric)         : {judged_learning:.4f}", flush=True)
    if flat_turns:
        print(f"  engagement dims  behavioral {_dim_mean(flat_turns,'behavioral'):.2f}"
              f" / affective {_dim_mean(flat_turns,'affective'):.2f}"
              f" / cognitive {_dim_mean(flat_turns,'cognitive'):.2f}"
              f" / learning_evidence {_dim_mean(flat_turns,'learning_evidence'):.2f}"
              f"   (n={len(flat_turns)} student turns)", flush=True)
    print(f"  learning dims    progress {_dim_mean(learning_scores,'solution_progress'):.2f}"
          f" / understanding {_dim_mean(learning_scores,'understanding'):.2f}"
          f" / misconception {_dim_mean(learning_scores,'misconception_repair', True):.2f}"
          f" / independence {_dim_mean(learning_scores,'independence'):.2f}", flush=True)
    print(f"  tutor_leaked flag rate (appendix)          : "
          f"{sum(s.get('tutor_leaked', False) for s in learning_scores)/len(learning_scores):.4f}", flush=True)
    print("--- Transfer test (sibling problem, same domain/difficulty) ---", flush=True)
    print(f"Sibling solve rate (cold, no dialogue)      : {transfer_control:.4f}", flush=True)
    print(f"Sibling solve rate (after dialogue, in-ctx) : {transfer_treatment:.4f}", flush=True)
    print(f"Transfer Delta                              : {transfer_delta:+.4f}", flush=True)
    print(
        f"Domain-match fallback (difficulty-only)     : {n_domain_fallback}/{n_problems}",
        flush=True,
    )

    # Split Delta Solve Rate by engaged vs disengaged subpopulation (per-conv;
    # valid because num_samples_per_problem is 1). Pins the causal mechanism:
    # do the students who leave actually fail to learn?
    eng, dis = [], []
    for c in conversations:
        pre_c, post_c = c.get_initial_rm_reward(), c.get_end_rm_reward()
        if pre_c is None or post_c is None:
            continue
        (dis if getattr(c, "disengaged", False) else eng).append(
            (pre_c, post_c, post_c - pre_c)
        )

    def _summ(rows):
        if not rows:
            return "n=0"
        n = len(rows)
        pre_m = sum(r[0] for r in rows) / n
        post_m = sum(r[1] for r in rows) / n
        d_m = sum(r[2] for r in rows) / n
        return f"n={n:3d}  pre={pre_m:.3f}  post={post_m:.3f}  Delta={d_m:+.4f}"

    print("--- Delta split by engagement ---", flush=True)
    print(f"  ENGAGED    : {_summ(eng)}", flush=True)
    print(f"  DISENGAGED : {_summ(dis)}", flush=True)
    print("============================================================\n", flush=True)

    # Full transcripts for inspection.
    for idx, c in enumerate(conversations):
        print(f"\n############### CONVERSATION {idx} ###############", flush=True)
        print(
            f"TYPE={c.type.name} | disengaged={getattr(c, 'disengaged', False)} | "
            f"turns={len(c.conversation)} | "
            f"pre={c.get_initial_rm_reward()} post={c.get_end_rm_reward()}",
            flush=True,
        )
        print(f"PROBLEM: {c.problem[:400]}", flush=True)
        print(f"GROUND TRUTH ANSWER: {c.answer}", flush=True)
        for m in c.conversation:
            print(f"\n--- {m['role'].upper()} ---", flush=True)
            print(m["content"][:900], flush=True)
        print("\n=== END CONVERSATION ===", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
