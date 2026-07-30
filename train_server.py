"""Rollout/reward server for engagement-conditioned RL training.

Same contract as vllm_server.py (the GRPO trainer's counterpart), but hosts
TrainEngagementClassroom: policy rollouts against UserLM-8b plus gpt-oss-20b
judge rewards (per-turn engagement, terminal learning outcome).

Endpoints: /sample_conversations, /get_engagement_reward, /get_learning_reward,
/wait_batch. Run inside .venv-eval with env_eval.sh sourced.
"""
import json
import os
import threading
import warnings
from typing import List, Optional

import hydra
import uvicorn
import wandb
from dotenv import load_dotenv
from fastapi import FastAPI
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pydantic import BaseModel

from config.train_rl_engagement_model import TrainEngagementRLConfig
from train_engagement_classroom import TrainEngagementClassroom
from train_osim_classroom import TrainOsimClassroom
from judge_rewards import learning_scalar
from src.classroom import Conversation
from src.utils.utils import init_logger

logger = init_logger()
warnings.filterwarnings("ignore")
load_dotenv()

lock = threading.Lock()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainEngagementRLConfig)

classroom: TrainEngagementClassroom = None
config: TrainEngagementRLConfig = None
app = FastAPI()


class ConversationSampleRequest(BaseModel):
    problems: List[str]
    answers: List[str]
    meta: dict = {}
    # Worked reference solutions, positionally aligned with `problems`. Shown
    # only to the teacher (see Conversation.reference_solution). Optional so
    # datasets without a solution column keep working unchanged.
    solutions: Optional[List[str]] = None


class RewardRequest(BaseModel):
    conversations: list[str]


@app.post("/sample_conversations")
def sample_conversations(request: ConversationSampleRequest):
    global classroom, config

    with lock:
        conversations = classroom.sample_conversations(
            problems=request.problems, answers=request.answers, meta=request.meta,
            solutions=request.solutions,
        )

    eng = [classroom.get_engagement_reward(c) for c in conversations]
    lea = [classroom.get_learning_reward(c) for c in conversations]
    # The three reward COMPONENTS, logged separately: with per-turn credit
    # assignment the trainer builds advantages directly and TRL's
    # rewards/<func> metrics no longer describe what is being optimised.
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0
    pt_eng, pt_evid = [], []
    for c in conversations:
        kept = [t for t in c.turn_scores if not t.get("role_drift", False)]
        pt_eng.append(_mean([(t["behavioral"] + t["affective"] + t["cognitive"]) / 12.0
                             for t in kept]))
        pt_evid.append(_mean([t.get("learning_evidence", 0) / 4.0 for t in kept]))
    term = [learning_scalar(getattr(c, "learning_scores", None)) for c in conversations]
    dis = [c.disengaged for c in conversations]
    drift_trunc = sum(c.role_drifted for c in conversations)
    drift_resamples = sum(c.drift_resamples for c in conversations)
    judge_drift = sum(
        1
        for c in conversations
        for t in c.turn_scores
        if t.get("role_drift", False)
    )
    leaked = sum(
        1
        for c in conversations
        if getattr(c, "learning_scores", {}).get("tutor_leaked", False)
    )

    # Persist every rollout batch (full turns + judge scores) for inspection.
    dump_dir = os.path.join(config.logging.save_dir, "dialogues")
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(
        dump_dir, f"batch_{len(classroom.conversation_sets):04d}.jsonl"
    )
    with open(dump_path, "w") as f:
        for c, e, l in zip(conversations, eng, lea):
            f.write(
                json.dumps(
                    {
                        "problem": c.problem,
                        "answer": c.answer,
                        "disengaged": c.disengaged,
                        "disengage_reason": c.disengage_reason,
                        "role_drifted": c.role_drifted,
                        "drift_resamples": c.drift_resamples,
                        "tutor_leaked": getattr(c, "learning_scores", {}).get(
                            "tutor_leaked", False
                        ),
                        "conversation": c.conversation,
                        "turn_scores": c.turn_scores,
                        "learning_scores": c.learning_scores,
                        "engagement_reward": e,
                        "learning_reward": l,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info(f"Dialogues dumped to {dump_path}")
    logger.info(
        f"Batch of {len(conversations)}: "
        f"engagement mean {sum(eng)/len(eng):.3f}, "
        f"learning mean {sum(lea)/len(lea):.3f}, "
        f"disengaged {sum(dis)}/{len(dis)}, "
        f"drift: {drift_resamples} resamples, {drift_trunc} truncated, "
        f"{judge_drift} judge-flagged turns, leaked {leaked}/{len(conversations)} | "
        f"components: per-turn eng {_mean(pt_eng):.3f}, "
        f"per-turn evidence {_mean(pt_evid):.3f}, terminal {_mean(term):.3f}"
    )

    if config.logging.wandb:
        # Full rollout transcripts go to the local dialogues/ dump above only
        # (not wandb) -- a per-batch wandb.Table of every conversation ran up
        # wandb's storage fast over a 200-step run. Scalar batch metrics still
        # get logged for the live training curves.
        wandb.log(
            {
                "server/engagement_reward_mean": sum(eng) / len(eng),
                "server/learning_reward_mean": sum(lea) / len(lea),
                "server/disengagement_rate": sum(dis) / len(dis),
                "server/drift_truncation_rate": drift_trunc / len(conversations),
                "server/drift_resamples": drift_resamples,
                "server/judge_flagged_drift_turns": judge_drift,
                "server/leak_rate": leaked / len(conversations),
                "server/per_turn_engagement_mean": _mean(pt_eng),
                "server/per_turn_learning_evidence_mean": _mean(pt_evid),
                "server/terminal_rubric_mean": _mean(term),
            }
        )

    return [c.get_trainable_representation() for c in conversations]


@app.post("/get_engagement_reward")
def get_engagement_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    return [classroom.get_engagement_reward(c) for c in conversations]


@app.post("/get_learning_reward")
def get_learning_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    return [classroom.get_learning_reward(c) for c in conversations]


@app.post("/get_per_turn_rewards")
def get_per_turn_rewards(request: RewardRequest):
    """Per-student-turn rewards, aligned with each dialogue's student turns.
    Returns a ragged list-of-lists; the trainer maps entry t onto the tokens of
    teacher turn t."""
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    return [classroom.get_per_turn_rewards(c) for c in conversations]


@app.get("/wait_batch")
def wait_batch():
    with lock:
        return {"message": "Batch has been run."}


@hydra.main(config_path="config/train_rl", version_base=None)
def main(cfg: TrainEngagementRLConfig):
    global classroom, config

    default_config = OmegaConf.structured(TrainEngagementRLConfig)
    cfg = OmegaConf.merge(default_config, cfg)
    config = cfg

    logger.info("Resolved config:\n" + OmegaConf.to_yaml(cfg))

    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.wandb_project + "-server",
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=cfg.logging.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )

    # simulator selects the in-dialogue student. Only the classroom changes;
    # the reward path is shared, so the two are a controlled pair.
    simulator = getattr(cfg, "simulator", "userlm")
    kwargs = dict(
        teacher_model_cfg=cfg.teacher_model,
        engagement_model_cfg=cfg.engagement_model,
        judge_model_cfg=cfg.judge_model,
        generation_cfg=cfg.generation,
        model_save_path=os.path.join(cfg.logging.save_dir, "policy"),
        leak_multiplier=cfg.reward.leak_multiplier,
        learning_weight=cfg.reward.learning_weight,
        engagement_weight=cfg.reward.engagement_weight,
        terminal_weight=cfg.reward.terminal_weight,
    )
    if simulator == "osim":
        logger.info(f"Simulator: OSIM (persona {cfg.persona_path})")
        classroom = TrainOsimClassroom(persona_path=cfg.persona_path, **kwargs)
    elif simulator == "userlm":
        logger.info("Simulator: UserLM-8b")
        classroom = TrainEngagementClassroom(**kwargs)
    else:
        raise ValueError(f"Unknown simulator {simulator!r}")

    uvicorn.run(app, host="0.0.0.0", port=cfg.generation.server_port)


if __name__ == "__main__":
    main()
