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
from typing import List

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


class RewardRequest(BaseModel):
    conversations: list[str]


@app.post("/sample_conversations")
def sample_conversations(request: ConversationSampleRequest):
    global classroom, config

    with lock:
        conversations = classroom.sample_conversations(
            problems=request.problems, answers=request.answers, meta=request.meta
        )

    eng = [classroom.get_engagement_reward(c) for c in conversations]
    lea = [classroom.get_learning_reward(c) for c in conversations]
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
        f"{judge_drift} judge-flagged turns, leaked {leaked}/{len(conversations)}"
    )

    if config.logging.wandb:
        df = classroom.to_pd_latest()
        df["engagement_reward"] = eng
        df["learning_reward"] = lea
        df["disengaged"] = dis
        df = df[
            [
                "Problem",
                "Answer",
                "Conversation",
                "engagement_reward",
                "learning_reward",
                "disengaged",
            ]
        ].astype(str)
        wandb.log(
            {
                f"batch_{len(classroom.conversation_sets)}": wandb.Table(dataframe=df),
                "server/engagement_reward_mean": sum(eng) / len(eng),
                "server/learning_reward_mean": sum(lea) / len(lea),
                "server/disengagement_rate": sum(dis) / len(dis),
                "server/drift_truncation_rate": drift_trunc / len(conversations),
                "server/drift_resamples": drift_resamples,
                "server/judge_flagged_drift_turns": judge_drift,
                "server/leak_rate": leaked / len(conversations),
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

    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.wandb_project + "-server",
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=cfg.logging.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )

    classroom = TrainEngagementClassroom(
        teacher_model_cfg=cfg.teacher_model,
        engagement_model_cfg=cfg.engagement_model,
        judge_model_cfg=cfg.judge_model,
        generation_cfg=cfg.generation,
        model_save_path=os.path.join(cfg.logging.save_dir, "policy"),
        leak_multiplier=cfg.reward.leak_multiplier,
    )

    uvicorn.run(app, host="0.0.0.0", port=cfg.generation.server_port)


if __name__ == "__main__":
    main()
