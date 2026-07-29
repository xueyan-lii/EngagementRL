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
    # Multiplier applied to the learning reward when the terminal judge flags
    # tutor leakage (tutor_leaked). 0.0 = leaking forfeits all learning credit.
    # 1.0 = gate OFF. Leakage is no longer a headline metric (it is TutorRL's
    # focus, not ours), and the terminal learning rubric's EVIDENCE RULE already
    # penalises tutor-stated answers per-dimension. The multiplicative gate on
    # top of that was firing on ~70% of dialogues lasting more than one turn
    # (measured over 3840 v3 training dialogues) and is anti-correlated with
    # engagement -- it zeroed learning precisely on the long, engaged dialogues
    # the objective is trying to produce. Leak rate is still recorded.
    leak_multiplier: float = 1.0


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
    # Which model drives the in-dialogue student turns: "userlm" or "osim".
    simulator: str = "userlm"
    # OSIM only: persona placed in the system slot alongside the problem.
    persona_path: str = "prompt_templates/personas/osim_passive.txt"
