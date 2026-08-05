"""Judge-only eval: dialogue-level ENGAGEMENT + LEARNING for a
(teacher x student-simulator) pair. No solve-rate probe, no sibling transfer.

Why a separate driver: eval_engagement.py's sample_conversations() also runs
the pre-dialogue attempt, the post-dialogue solution and the transfer test,
all of which need the Llama knowledge probe and dominate the runtime. Those
metrics are not wanted here, so this stops after the dialogue and scores it
with the SAME rubrics the training reward uses (judge_rewards.judge_dialogues).
`generation.skip_student_model=true` means the probe is never even loaded --
three models instead of four.

Everything that shapes the dialogue is unchanged: the same Conversation state
machine, teacher prompt, personas, max_turns, and the same simulator classes
(EngagementClassroom / OsimEngagementClassroom), so transcripts are comparable
with the full eval.

Reported per dialogue, macro-averaged over problems:
  - engagement: summed per-turn (behavioral+affective+cognitive)/12 normalised
    by max_turns//2, i.e. the capacity of the episode, not the turns taken.
  - learning:   the 4-dim terminal rubric.
  - leak flag rate: recorded for the appendix, not gated on.

Usage (single cell):
  python eval_judged.py --config-name judged \\
      teacher_model.model_name_or_path=Qwen/Qwen3-8B \\
      simulator=osim persona_path=prompt_templates/personas/osim_disengaged.txt \\
      run_name=qwen3-8b_osim-passive

Sweep over all combinations: ./run_judged_sweep.sh
"""
import json
import os
import warnings
from dataclasses import dataclass, field

import hydra
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config.eval import EvalConfig
from config.train_rl_model import StudentModelConfig
from engagement_classroom import EngagementClassroom
from judge_rewards import (LEARNING_DIMS, engagement_scalar, judge_dialogues,
                           learning_scalar, per_turn_rewards)
from osim_classroom import OsimEngagementClassroom
from src.classroom import ConversationState, ConversationType
from src.utils.utils import init_logger
from utils.data import load_datasets

load_dotenv()
logger = init_logger()
warnings.filterwarnings("ignore")


@dataclass
class JudgedEvalConfig(EvalConfig):
    engagement_model: StudentModelConfig = field(
        default_factory=lambda: StudentModelConfig(
            model_name_or_path="microsoft/UserLM-8b"
        )
    )
    simulator: str = "userlm"           # "userlm" | "osim"
    persona_path: str = "prompt_templates/personas/osim_disengaged.txt"
    run_name: str = "judged"
    out_dir: str = "logs/judged"


cs = ConfigStore.instance()
cs.store(name="config", node=JudgedEvalConfig)


def _macro_mean(values, n_problems, n_samples):
    out = []
    for i in range(n_problems):
        cur = [v for v in values[i * n_samples:(i + 1) * n_samples] if v is not None]
        if cur:
            out.append(sum(cur) / len(cur))
    return sum(out) / len(out) if out else float("nan")


def _dim_mean(scores, dim, skip_negative=False):
    vals = [s[dim] for s in scores if not (skip_negative and s[dim] < 0)]
    return sum(vals) / len(vals) if vals else float("nan")


@hydra.main(config_path="config/eval", version_base=None)
def main(cfg):
    cfg = OmegaConf.merge(OmegaConf.structured(JudgedEvalConfig), cfg)

    cls = OsimEngagementClassroom if cfg.simulator == "osim" else EngagementClassroom
    kwargs = {"persona_path": cfg.persona_path} if cfg.simulator == "osim" else {}
    logger.info(f"[{cfg.run_name}] teacher={cfg.teacher_model.model_name_or_path} "
                f"simulator={cfg.simulator} ({cfg.engagement_model.model_name_or_path})")
    classroom = cls(
        cfg.student_model, cfg.teacher_model, cfg.judge_model, cfg.reward_model,
        cfg.generation, None, cfg.engagement_model, **kwargs,
    )

    _, eval_data = load_datasets(cfg.dataset, cfg.seed)
    problems, answers = eval_data["problem"], eval_data["answer"]
    solutions = eval_data["solution"] if "solution" in eval_data.column_names else None
    n_problems, n_samples = len(problems), cfg.num_samples_per_problem

    sp, sa, ss = [], [], []
    for i in range(n_problems):
        sp += [problems[i]] * n_samples
        sa += [answers[i]] * n_samples
        if solutions is not None:
            ss += [solutions[i]] * n_samples

    # ---- dialogue only -------------------------------------------------
    from src.classroom import Conversation
    forced = (ConversationType.GUIDED
              if cfg.generation.forced_conversation_type == "guided" else None)
    sols = ss if solutions is not None else [None] * len(sp)
    conversations = []
    for p, a, s in zip(sp, sa, sols):
        c = Conversation(p, a, cfg.generation, forced, reference_solution=s)
        c.disengaged = False
        c.disengage_reason = None
        conversations.append(c)
    for c in conversations:
        c.start_conversation()

    turn_i = 1
    while any(c.state in (ConversationState.TEACHER_TURN,
                          ConversationState.STUDENT_TURN) for c in conversations):
        for state in (ConversationState.TEACHER_TURN, ConversationState.STUDENT_TURN):
            todo = [c for c in conversations if c.state == state]
            if not todo:
                continue
            who = "Teacher" if state == ConversationState.TEACHER_TURN else "Student"
            logger.info(f"turn {turn_i}: {who} ({len(todo)} convs)")
            if state == ConversationState.TEACHER_TURN:
                classroom.generate_next_teacher_utterances(todo, {})
            else:
                classroom.generate_next_student_utterances(todo)
            turn_i += 1

    # ---- score with the training rubrics --------------------------------
    logger.info("Scoring dialogues (engagement + learning)...")

    def run_batch(messages):
        return [r.outputs[0].text for r in
                classroom.judge_model.run_batch(messages,
                                                classroom.sampling_params_judge)]

    dialogues = [c._get_hidden_conversation() for c in conversations]
    turn_scores, learning_turn_scores = judge_dialogues(run_batch, dialogues, sp, sa)

    max_student_turns = max(1, cfg.generation.max_turns // 2)
    eng = [engagement_scalar(t, max_student_turns) for t in turn_scores]
    # Learning is per-turn now; report the dialogue mean over its student turns.
    learn = [
        (sum(learning_scalar(x) for x in lts) / len(lts)) if lts else 0.0
        for lts in learning_turn_scores
    ]
    flat = [t for ts in turn_scores for t in ts if not t.get("role_drift")]
    flat_learn = [x for lts in learning_turn_scores for x in lts if x]

    res = {
        "run_name": cfg.run_name,
        "teacher": cfg.teacher_model.model_name_or_path,
        "simulator": cfg.simulator,
        "simulator_model": cfg.engagement_model.model_name_or_path,
        "persona": cfg.persona_path if cfg.simulator == "osim" else None,
        "judge": cfg.judge_model.model_name_or_path,
        "n_problems": n_problems, "n_samples": n_samples,
        "engagement": _macro_mean(eng, n_problems, n_samples),
        "learning": _macro_mean(learn, n_problems, n_samples),
        "behavioral": _dim_mean(flat, "behavioral"),
        "affective": _dim_mean(flat, "affective"),
        "cognitive": _dim_mean(flat, "cognitive"),
        "solution_progress": _dim_mean(flat_learn, "solution_progress"),
        "understanding": _dim_mean(flat_learn, "understanding"),
        "misconception_repair": _dim_mean(flat_learn, "misconception_repair"),
        "leak_flag_rate": sum(s.get("tutor_leaked", False)
                              for s in flat_learn) / max(1, len(flat_learn)),
        "disengage_rate": sum(getattr(c, "disengaged", False)
                              for c in conversations) / len(conversations),
        "avg_dialogue_msgs": sum(len(c.conversation)
                                 for c in conversations) / len(conversations),
        "n_student_turns_scored": len(flat),
    }

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, f"{cfg.run_name}.json"), "w") as f:
        json.dump({"summary": res, "per_dialogue": [
            {"engagement": e, "learning": l, "leaked": s.get("tutor_leaked", False),
             "disengaged": getattr(c, "disengaged", False),
             "n_msgs": len(c.conversation)}
            for e, l, s, c in zip(eng, learn, learning_turn_scores, conversations)]}, f)
    with open(os.path.join(cfg.out_dir, f"{cfg.run_name}_dialogues.jsonl"), "w") as f:
        for c, e, l in zip(conversations, eng, learn):
            f.write(json.dumps({"problem": c.problem, "answer": c.answer,
                                "engagement": e, "learning": l,
                                "conversation": c._get_hidden_conversation()}) + "\n")

    print(f"\n========== {cfg.run_name} ==========")
    for k in ("teacher", "simulator_model", "persona", "judge"):
        print(f"  {k:<18}: {res[k]}")
    print(f"  ENGAGEMENT        : {res['engagement']:.4f}")
    print(f"  LEARNING          : {res['learning']:.4f}")
    print(f"    beh {res['behavioral']:.2f} / aff {res['affective']:.2f} / "
          f"cog {res['cognitive']:.2f}")
    print(f"    progress {res['solution_progress']:.2f} / underst {res['understanding']:.2f} / "
          f"miscon {res['misconception_repair']:.2f}")
    print(f"  disengage rate    : {res['disengage_rate']:.4f}")
    print(f"  avg dialogue msgs : {res['avg_dialogue_msgs']:.1f}")
    print(f"  leak flag (appendix): {res['leak_flag_rate']:.4f}")
    print(f"-> {cfg.out_dir}/{cfg.run_name}.json")


if __name__ == "__main__":
    main()
