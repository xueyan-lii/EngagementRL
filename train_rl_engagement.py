"""GRPO training entry point for engagement-conditioned tutoring.

Adapted from train_rl.py; differences:
  - config class TrainEngagementRLConfig (adds engagement_model + reward weights)
  - rewards = judge-based engagement + learning (served by train_server.py)
    instead of solve-rate/thinking/end/length
  - attn_implementation = sdpa (no flash-attn build needed on Blackwell)
"""
from datetime import timedelta

import os
import warnings

import hydra
import requests
import torch
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import Dataset
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from config.train_rl_engagement_model import TrainEngagementRLConfig
from src.grpo.config import ClassroomGRPOConfig
from src.grpo.trainer import ClassroomGRPOTrainer
from src.utils.utils import init_logger
from utils.data import load_datasets

warnings.filterwarnings("ignore")
load_dotenv()

logger = init_logger()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainEngagementRLConfig)


def construct_judge_reward_func(endpoint: str, server_port: int, weight: float):
    def reward_func(completions, **kwargs):
        response = requests.post(
            f"http://localhost:{server_port}/{endpoint}",
            json={"conversations": completions},
        )
        response.raise_for_status()
        return [weight * r for r in response.json()]

    reward_func.__name__ = endpoint
    return reward_func


@hydra.main(config_path="config/train_rl", version_base=None)
def main(cfg: TrainEngagementRLConfig):

    default_config = OmegaConf.structured(TrainEngagementRLConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    model_config = cfg.teacher_model
    train_config = cfg.train
    logging_config = cfg.logging
    data_config = cfg.dataset

    set_seed(cfg.seed)

    kwargs = [InitProcessGroupKwargs(timeout=timedelta(hours=10))]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    if accelerator.is_main_process:
        logger.info("Resolved config:\n" + OmegaConf.to_yaml(cfg))

    if logging_config.wandb and accelerator.is_main_process:
        wandb.init(
            project=logging_config.wandb_project,
            name=logging_config.wandb_run_name,
            entity=logging_config.wandb_entity,
            group=logging_config.run_group,
            tags=logging_config.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )
    accelerator.wait_for_everyone()

    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation=train_config.attn_implementation,
        torch_dtype=torch.bfloat16,
        use_cache=False if train_config.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=True
    )

    logger.info(f"Loading datasets from {data_config.train_datasets}")
    train_dataset, _ = load_datasets(data_config, cfg.seed)
    logger.info(f"Loaded {len(train_dataset)} training examples")

    train_dataset: Dataset = train_dataset.map(
        lambda ex: {"prompt": ex["problem"], "answer": ex["answer"]},
        num_proc=4,
        desc="Applying template",
    )

    engagement_reward = construct_judge_reward_func(
        "get_engagement_reward",
        cfg.generation.server_port,
        cfg.reward.engagement_weight,
    )
    learning_reward = construct_judge_reward_func(
        "get_learning_reward",
        cfg.generation.server_port,
        cfg.reward.learning_weight,
    )

    trainer = ClassroomGRPOTrainer(
        model=model_config.model_name_or_path,
        reward_funcs=[learning_reward, engagement_reward],
        args=ClassroomGRPOConfig(
            gradient_accumulation_steps=cfg.train.num_samples_per_problem
            * cfg.train.number_of_problems_per_batch
            // cfg.train.per_device_train_batch_size
            // accelerator.num_processes,
            gradient_checkpointing=train_config.gradient_checkpointing,
            num_generations=cfg.train.num_samples_per_problem,
            per_device_train_batch_size=cfg.train.per_device_train_batch_size,
            num_iterations=cfg.train.mu,
            epsilon=cfg.train.epsilon,
            beta=cfg.train.beta,
            learning_rate=cfg.train.learning_rate,
            optim=cfg.train.optimizer,
            bf16=True,
            run_name=cfg.logging.wandb_run_name,
            model_init_kwargs=model_kwargs,
            hub_model_id=cfg.huggingface.name,
            hub_private_repo=False,
            report_to=["wandb"] if logging_config.wandb else [],
            save_strategy="steps",
            lr_scheduler_type=train_config.lr_scheduler_type,
            num_train_epochs=train_config.epochs,
            max_steps=train_config.max_steps,
            max_completion_length=model_config.vllm.max_length,
            logging_steps=1,
            save_steps=cfg.logging.save_steps,
            save_on_each_node=False,
            save_only_model=False,
            save_total_limit=2,  # 114GB each (deepspeed optim state); disk-full at 3+
            output_dir=cfg.logging.save_dir,
            max_grad_norm=1.0,
            temperature=cfg.teacher_model.vllm.temperature,
            vllm_server_port=cfg.generation.server_port,
            use_experimental_shared_memory=cfg.generation.use_experimental_shared_memory,
            batch_size_reference_model=cfg.train.batch_size_ref_model,
            # Dr.GRPO: raw advantages. std-normalization turned all-zero-reward
            # batches into amplified noise (v0 collapse at ~step 95).
            scale_rewards=False,
            save_policy_to_disk_every_n_steps=cfg.train.save_policy_to_disk_every_n,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    last_ckpt = None
    if os.path.isdir(cfg.logging.save_dir):
        last_ckpt = get_last_checkpoint(cfg.logging.save_dir)
        logger.info(f"Last checkpoint: {last_ckpt}")

    logger.info("Training...")
    train_results = trainer.train(resume_from_checkpoint=last_ckpt)
    logger.info("Training complete!")
    logger.info(train_results)

    trainer.model.config.use_cache = True
    trainer.save_model(logging_config.save_dir + "/model")

    if cfg.huggingface.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
