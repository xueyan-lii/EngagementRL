from dataclasses import dataclass, field

from config.train_rl_model import (
    RLModelTrainingConfig,
    StudentModelConfig,
    JudgeModelConfig,
)


@dataclass
class EngagementRewardConfig:
    # R = learning_weight * learning + engagement_weight * engagement.
    # With GRPO group-normalization only the ratio matters (the Pareto knob).
    engagement_weight: float = 0.5
    learning_weight: float = 1.0


@dataclass
class TrainEngagementRLConfig(RLModelTrainingConfig):
    """RL training config with a UserLM engagement simulator and judge-based
    rewards. `student_model` / `reward_model` from the base config are unused
    at train time (kept so shared code paths don't break)."""

    engagement_model: StudentModelConfig = field(
        default_factory=lambda: StudentModelConfig(
            model_name_or_path="microsoft/UserLM-8b"
        )
    )
    reward: EngagementRewardConfig = field(default_factory=EngagementRewardConfig)
