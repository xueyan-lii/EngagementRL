"""EngagementClassroom: TutorRL eval where the in-dialogue STUDENT turns are
driven by a disengagement-capable user simulator (microsoft/UserLM-8b) instead
of the always-engaged Llama student.

Design (see ../BASELINE_BUILD_PLAN.md):
  - Engagement simulator (UserLM) generates the student's conversational turns
    and may emit <|endconversation|> to leave.  ONLY this is swapped.
  - Knowledge probe stays the base-class `student_model` (bf16 Llama): it computes
    the pre-dialog initial attempt and the post-dialog final solution (boxed,
    math-verify), so Delta Solve Rate is identical/comparable to the paper.
  - When UserLM disengages we terminate the dialog (-> JUDGE_TURN) and score the
    transcript-so-far (partial-credit probe).
"""
import re

from transformers import AutoTokenizer
from vllm import SamplingParams

from src.classroom import Classroom, Conversation, ConversationState
from src.vllm.data_parallel_vllm import ParallelvLLMInference

END_CONV_TOKEN = "<|endconversation|>"


def _clean_for_userlm(content: str) -> str:
    """Hide tutor thinking tags / end markers before showing a turn to UserLM."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    return content.replace("<end_of_conversation>", "").strip()


class EngagementClassroom(Classroom):
    def __init__(
        self,
        student_model_cfg,
        teacher_model_cfg,
        judge_model_cfg,
        reward_model_cfg,
        generation_cfg,
        model_save_path,
        engagement_model_cfg,
        log_file_path: str = None,
    ):
        super().__init__(
            student_model_cfg,
            teacher_model_cfg,
            judge_model_cfg,
            reward_model_cfg,
            generation_cfg,
            model_save_path,
            log_file_path,
        )
        self.engagement_model_cfg = engagement_model_cfg
        self.engagement_model = ParallelvLLMInference(
            model_path=engagement_model_cfg.model_name_or_path,
            gpu_memory_utilization=engagement_model_cfg.vllm.gpu_memory_utilization,
            gpu_ids=engagement_model_cfg.vllm.gpu_ids,
            max_model_len=engagement_model_cfg.vllm.max_length,
            max_num_seqs=engagement_model_cfg.vllm.max_num_seqs,
            model_save_path=None,
            load_and_unload=False,      # resident; UserLM stays on its own GPU
            enable_sleep_mode=False,
            enforce_eager=True,
            use_v0=False,
            userlm_mode=True,           # render template + BOS, use .generate()
        )
        self.engagement_tokenizer = AutoTokenizer.from_pretrained(
            engagement_model_cfg.model_name_or_path
        )
        # UserLM ends a turn at <|eot_id|> and leaves at <|endconversation|>;
        # both must stop generation (mirrors serve_user.py's eos_token_id), else
        # vllm runs to max_tokens and the model drifts into assistant-style text.
        self.eot_id = self.engagement_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.endconv_id = self.engagement_tokenizer.convert_tokens_to_ids(
            END_CONV_TOKEN
        )
        self.sampling_params_engagement = SamplingParams(
            temperature=engagement_model_cfg.vllm.temperature,
            top_p=engagement_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
            stop_token_ids=[self.eot_id, self.endconv_id],
        )

    # Generic intent (in-distribution for UserLM: a short goal, not the pasted
    # problem). The problem reaches UserLM via a synthetic student-first opening.
    INTENT = "You are a student trying to solve a math problem."

    def _opening_turn(self, conv: Conversation) -> str:
        """Synthetic first student turn for UserLM's view only: the student
        presents the problem (WildChat-style). The teacher never sees this; it
        initiates from its own system prompt, which already holds the problem."""
        return f"Here's a problem I'm trying to solve:\n\n{conv.problem}\n\nCan you help me?"

    def _build_userlm_messages(self, conv: Conversation):
        """UserLM's view is student-first and in-distribution: [generic intent,
        synthetic 'here's the problem' opening, tutor turn, ...]. The canonical
        conversation is teacher-first (GUIDED), so the teacher stays in its own
        in-distribution (tutor-initiation) setting. Same problem info for both."""
        messages = [
            {"role": "system", "content": self.INTENT},
            {"role": "user", "content": self._opening_turn(conv)},
        ]
        for m in conv.conversation:  # canonical (GUIDED): teacher turn first
            role = "user" if m["role"] == "student" else "assistant"
            messages.append({"role": role, "content": _clean_for_userlm(m["content"])})
        return messages

    def generate_next_student_utterances(self, conversations):
        # Canonical conversation is GUIDED (teacher already spoke first), so every
        # student turn here is a reaction to a tutor turn -> UserLM may engage or
        # emit <|endconversation|>. Its view gets the synthetic opening prepended.
        prompts = [self._build_userlm_messages(c) for c in conversations]
        responses = self.engagement_model.run_batch(
            prompts, self.sampling_params_engagement
        )
        utterances = []
        for conv, response in zip(conversations, responses):
            out = response.outputs[0]
            raw = out.text
            clean = raw.replace(END_CONV_TOKEN, "").strip()
            explicit_end = (
                out.stop_reason == self.endconv_id
                or self.endconv_id in (out.token_ids or [])
                or END_CONV_TOKEN in raw
            )
            # A blank turn (UserLM ends its turn with no content) is treated as
            # disengagement too: a student who goes silent has checked out.
            ended = explicit_end or clean == ""
            conv.add_message(clean)  # appends student turn, -> TEACHER_TURN
            if ended:
                conv.disengaged = True
                conv.disengage_turn = len(conv.conversation)
                conv.disengage_reason = "endconversation" if explicit_end else "silence"
                conv.state = ConversationState.JUDGE_TURN  # student left
            utterances.append(clean)
        return utterances
