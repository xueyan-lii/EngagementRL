"""TrainOsimClassroom: the GRPO rollout/reward environment with OdysSim
(cmu-lti/osim-8b) as the in-dialogue student instead of UserLM-8b.

Subclasses TrainEngagementClassroom and overrides only the simulator-specific
pieces, so the reward path (per-turn engagement judge, terminal learning judge,
participation gate, leak flag, dialogue dumps) stays byte-identical between the
two simulators and the pair is a controlled comparison.

Four differences, all forced by how OSIM works (see PROMPTS.md):

1. ROLE MAPPING IS INVERTED. UserLM occupies the `user` role with the tutor as
   `assistant`. OSIM imitates the human but *generates* the `assistant` role,
   so the tutor becomes `user` and the student's own turns become `assistant`.

2. THE PROBLEM ENTERS VIA THE SYSTEM SLOT, not a synthetic student-first
   opening. UserLM needed that opening to stay in its student-first training
   distribution; OSIM does not, because a GUIDED dialogue is already
   teacher-first. But the problem still has to reach the student: measured,
   relying on the tutor to restate it drops grounding to 0.21 (the student
   invents its own problem) because the tutor fully restates only 26% of the
   time.

3. NO TERMINATION TOKEN. OSIM's vocabulary is stock Qwen3; P(<|im_end|>) at the
   next-turn position is ~1e-11, so it essentially never ends a conversation.
   Disengagement is therefore read from an explicit textual sign-off. That
   proxy is conservative and UNCALIBRATED -- it is not comparable to UserLM's
   p_end, and the rate will be far lower. Engagement is meant to be carried by
   the graded judge score, not by this binary.

4. `userlm_mode=False`. That flag exists only to prepend the BOS UserLM's chat
   template omits; OSIM is stock Qwen3 ChatML and takes the normal chat path.

Assistant-drift resampling is inherited unchanged. The drift regex was tuned on
UserLM's Llama idioms and fires on 0/1920 OSIM turns measured so far, which may
mean OSIM does not drift in early turns or may mean the regex is blind to its
idioms -- the judge's `role_drift` flag is the backstop either way.
"""
import os

from transformers import AutoTokenizer
from vllm import SamplingParams

from src.classroom import Conversation, ConversationState
from src.vllm.data_parallel_vllm import ParallelvLLMInference
from engagement_classroom import _clean_for_userlm
from osim_classroom import looks_like_signoff
from train_engagement_classroom import TrainEngagementClassroom, looks_like_assistant

import logging

logger = logging.getLogger(__name__)

DEFAULT_PERSONA = "prompt_templates/personas/osim_disengaged.txt"


class TrainOsimClassroom(TrainEngagementClassroom):
    def __init__(self, *args, persona_path: str = DEFAULT_PERSONA, **kwargs):
        # Persona must exist before super().__init__ runs, because the engine
        # setup hook it calls is overridden below.
        with open(persona_path) as f:
            self.persona = f.read().strip()
        self.persona_path = persona_path
        super().__init__(*args, **kwargs)
        logger.info(f"TrainOsimClassroom: persona={persona_path}")

    # ------------------------------------------------------------------
    # Simulator setup (hooks from the parent constructor)
    # ------------------------------------------------------------------

    def _setup_engagement_model(self, engagement_model_cfg):
        self.engagement_model = ParallelvLLMInference(
            model_path=engagement_model_cfg.model_name_or_path,
            gpu_memory_utilization=engagement_model_cfg.vllm.gpu_memory_utilization,
            gpu_ids=engagement_model_cfg.vllm.gpu_ids,
            max_model_len=engagement_model_cfg.vllm.max_length,
            max_num_seqs=engagement_model_cfg.vllm.max_num_seqs,
            model_save_path=None,
            load_and_unload=False,
            enable_sleep_mode=False,
            enforce_eager=True,
            use_v0=False,
            userlm_mode=False,  # stock Qwen3 ChatML -> normal chat path
        )
        self.engagement_tokenizer = AutoTokenizer.from_pretrained(
            engagement_model_cfg.model_name_or_path
        )
        # No conversation-end token exists; only <|im_end|> ends a turn, and
        # vLLM stops on it via the tokenizer's eos by default.
        self.eot_id = self.engagement_tokenizer.eos_token_id
        self.endconv_id = None

    def _build_engagement_sampling_params(self, engagement_model_cfg, generation_cfg):
        return SamplingParams(
            temperature=engagement_model_cfg.vllm.temperature,
            top_p=engagement_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
        )

    # ------------------------------------------------------------------
    # OSIM's view of the dialogue
    # ------------------------------------------------------------------

    def _build_osim_messages(self, conv: Conversation):
        system = (
            f"{self.persona}\n\nThe problem you are working on:\n\n{conv.problem}"
        )
        messages = [{"role": "system", "content": system}]
        for m in conv.conversation:
            role = "assistant" if m["role"] == "student" else "user"
            messages.append({"role": role, "content": _clean_for_userlm(m["content"])})
        return messages

    def _generate_student_turns(self, conversations):
        """Same rejection-sampling structure as the parent, with the UserLM
        end-token logic replaced by textual sign-off detection."""
        pending = list(conversations)
        for attempt in range(self.MAX_DRIFT_RESAMPLES + 1):
            if not pending:
                break
            prompts = [self._build_osim_messages(c) for c in pending]
            responses = self.engagement_model.run_batch(
                prompts, self.sampling_params_engagement
            )
            retry = []
            for conv, response in zip(pending, responses):
                clean = response.outputs[0].text.strip()
                # A blank turn is silence; an explicit sign-off is leaving.
                ended = clean == "" or looks_like_signoff(clean)
                if not ended and looks_like_assistant(clean):
                    conv.drift_resamples += 1
                    if attempt < self.MAX_DRIFT_RESAMPLES:
                        retry.append(conv)
                    else:
                        conv.role_drifted = True
                        conv.state = ConversationState.END
                        logger.warning(
                            "Persistent assistant-drift after "
                            f"{self.MAX_DRIFT_RESAMPLES} resamples; truncating "
                            f"dialogue (problem: {conv.problem[:60]!r})"
                        )
                    continue
                conv.add_message(clean)
                if ended:
                    conv.disengaged = True
                    conv.disengage_reason = "silence" if clean == "" else "signoff"
                    conv.state = ConversationState.END
            pending = retry
