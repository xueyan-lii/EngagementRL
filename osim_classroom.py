"""OsimEngagementClassroom: the eval classroom with OdysSim (cmu-lti/osim-8b)
driving the in-dialogue student turns instead of UserLM-8b.

Subclasses EngagementClassroom rather than editing it, so the validated UserLM
path stays byte-identical and the two simulators can be run as a controlled
pair (only the in-dialogue student changes).

Three things differ from UserLM:

1. ROLE MAPPING IS INVERTED. UserLM sits in the `user` role with the tutor as
   `assistant`. OSIM imitates the human but *generates* the `assistant` role,
   so the tutor becomes `user` and the simulated student's own turns become
   `assistant`.

2. THE PROBLEM ENTERS VIA THE SYSTEM PROMPT (design A, see PROMPTS.md). UserLM
   received it through a synthetic student-first opening turn, needed to keep
   UserLM in its student-first training distribution. OSIM has no such
   constraint -- a GUIDED dialogue is already teacher-first, matching OSIM's
   native order -- but the problem still has to reach it. Measured: relying on
   the tutor to restate the problem drops grounding to 0.21 (the student
   invents its own problem), because the tutor fully restates it only 26% of
   the time.

3. THERE IS NO TERMINATION TOKEN. UserLM emits `<|endconversation|>`, a
   supervised, calibrated action. OSIM's vocabulary is stock Qwen3 with no such
   token, and P(<|im_end|>) at the next-turn position is ~1e-11 -- it
   essentially never ends a conversation. Disengagement therefore has to be
   read out of the TEXT (an explicit sign-off), which is a conservative,
   UNCALIBRATED proxy and is NOT comparable to UserLM's p_end. Expect a much
   lower disengagement rate; that is a property of the simulator, not of the
   tutor being evaluated.
"""
import re

from transformers import AutoTokenizer
from vllm import SamplingParams

from src.classroom import Classroom, Conversation, ConversationState
from src.vllm.data_parallel_vllm import ParallelvLLMInference
from engagement_classroom import EngagementClassroom, _clean_for_userlm

# Explicit conversation-ending sign-offs. Deliberately conservative: it must
# fire on "I'm done here, thanks" but not on a mid-dialogue "thanks!" that is
# followed by more work. Requires the turn to be short AND to end on the
# sign-off, since OSIM's disengagement shows up as a brief closing message.
_SIGNOFF_RE = re.compile(
    r"("
    r"good\s?bye|\bbye\b|see you( later)?|have a (great|good|nice) (day|one)"
    r"|i'?m (all )?(done|good|set)( here| now)?"
    r"|that'?s (all|it)( i needed| for now| for today)?"
    r"|i'?ll (stop|leave) (here|it there)"
    r"|thanks for (your |the )?(help|time)[.!]?$"
    r"|i have to (go|run)|gotta (go|run)"
    r")",
    re.IGNORECASE,
)
# A sign-off inside a long turn is almost always politeness wrapped around real
# work, not departure.
MAX_SIGNOFF_CHARS = 220


def looks_like_signoff(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > MAX_SIGNOFF_CHARS:
        return False
    return bool(_SIGNOFF_RE.search(t))


class OsimEngagementClassroom(EngagementClassroom):
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
        persona_path: str = "prompt_templates/personas/osim_passive.txt",
    ):
        # Skip EngagementClassroom.__init__ (it hard-wires UserLM's BOS shim,
        # end-token ids and stop_token_ids); go straight to the base class for
        # the shared teacher/probe/judge setup.
        Classroom.__init__(
            self,
            student_model_cfg,
            teacher_model_cfg,
            judge_model_cfg,
            reward_model_cfg,
            generation_cfg,
            model_save_path,
            log_file_path,
        )
        self.engagement_model_cfg = engagement_model_cfg
        with open(persona_path) as f:
            self.persona = f.read().strip()

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
            # False, not True: userlm_mode exists only to prepend the BOS that
            # UserLM's chat template omits. OSIM is stock Qwen3 ChatML and
            # takes the normal llm.chat path.
            userlm_mode=False,
        )
        self.engagement_tokenizer = AutoTokenizer.from_pretrained(
            engagement_model_cfg.model_name_or_path
        )
        # Only <|im_end|> ends a turn; there is no conversation-end token.
        self.sampling_params_engagement = SamplingParams(
            temperature=engagement_model_cfg.vllm.temperature,
            top_p=engagement_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
        )

    def _build_osim_messages(self, conv: Conversation):
        """OSIM's view: persona + problem in the system slot, then the real
        dialogue with the tutor as `user` and the student as `assistant`."""
        system = (
            f"{self.persona}\n\nThe problem you are working on:\n\n{conv.problem}"
        )
        messages = [{"role": "system", "content": system}]
        for m in conv.conversation:
            role = "assistant" if m["role"] == "student" else "user"
            messages.append({"role": role, "content": _clean_for_userlm(m["content"])})
        return messages

    def generate_next_student_utterances(self, conversations):
        prompts = [self._build_osim_messages(c) for c in conversations]
        responses = self.engagement_model.run_batch(
            prompts, self.sampling_params_engagement
        )
        utterances = []
        for conv, response in zip(conversations, responses):
            clean = response.outputs[0].text.strip()
            # No end token exists, so disengagement is read from the text. A
            # blank turn still counts as going silent, exactly as for UserLM.
            ended = clean == "" or looks_like_signoff(clean)
            conv.add_message(clean)
            if ended:
                conv.disengaged = True
                conv.disengage_turn = len(conv.conversation)
                conv.disengage_reason = "silence" if clean == "" else "signoff"
                conv.state = ConversationState.JUDGE_TURN
            utterances.append(clean)
        return utterances
