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
from src.classroom import JudgeDecision
from utils.data import load_datasets
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


@hydra.main(config_path="config/eval", version_base=None)
def main(cfg):
    cfg = OmegaConf.merge(OmegaConf.structured(EngagementEvalConfig), cfg)

    classroom = EngagementClassroom(
        cfg.student_model,
        cfg.teacher_model,
        cfg.judge_model,
        cfg.reward_model,
        cfg.generation,
        None,
        cfg.engagement_model,
    )

    _, eval_data = load_datasets(cfg.dataset, cfg.seed)
    problems, answers = eval_data["problem"], eval_data["answer"]
    n_problems = len(problems)
    n_samples = cfg.num_samples_per_problem

    sample_problems, sample_answers = [], []
    for i in range(n_problems):
        sample_problems.extend([problems[i]] * n_samples)
        sample_answers.extend([answers[i]] * n_samples)

    logger.info("Sampling conversations (UserLM student)...")
    conversations = classroom.sample_conversations(
        sample_problems, sample_answers, compute_initial_attempt=True
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
