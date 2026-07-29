"""TrainEngagementClassroom: rollout + reward environment for
engagement-conditioned GRPO training.

Cast (all resident, one GPU each via gpu_ids):
  - teacher  = the policy being trained (vllm copy, reloads from
               save_dir/policy/checkpoint-N whenever the trainer writes one)
  - student  = microsoft/UserLM-8b (engagement simulator; may emit
               <|endconversation|> or a blank turn -> disengagement, dialogue ends)
  - judge    = openai/gpt-oss-20b scoring (a) each student turn on the 4-dim
               per-turn rubric and (b) the whole dialogue on the 4-dim
               learning-outcome rubric.

Rewards exposed per conversation, both in [0,1]:
  - engagement reward: mean over student turns of (beh+aff+cog)/12
  - learning reward:   mean of applicable learning dims / 4

Differences vs. the eval EngagementClassroom: no knowledge probe (no
pre/post solve rate at train time), no leakage/pedagogy judges, no reward
model. Inherits Classroom only for conversation bookkeeping helpers
(get_conversation_by_text, to_pd_latest); __init__ is fully replaced.
"""
import gc
import re
import time

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from src.classroom import (
    Classroom,
    Conversation,
    ConversationState,
    ConversationType,
)
from src.vllm.data_parallel_vllm import ParallelvLLMInference
from engagement_classroom import END_CONV_TOKEN, _clean_for_userlm
from judge_rewards import (
    DEFAULT_ENGAGEMENT_SCORES,
    DEFAULT_LEARNING_SCORES,
    ENGAGEMENT_SYSTEM_PROMPT,
    LEARNING_SYSTEM_PROMPT,
    ZERO_ENGAGEMENT_SCORES,
    build_engagement_user_prompt,
    build_learning_user_prompt,
    engagement_scalar,
    extract_engagement_scores,
    extract_learning_scores,
    learning_scalar,
)
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UserLM assistant-drift detection (known failure mode: the user simulator
# slips into assistant persona and starts tutoring/solving, which inflates
# engagement + learning-evidence rewards). Conservative markers only: a false
# positive merely costs one resample of a stochastic turn.
# ---------------------------------------------------------------------------
# Assistant-style openers (must appear at the very start of the turn).
_DRIFT_OPEN_RE = re.compile(
    r"^(certainly[!,]"
    r"|sure[,!] let'?s"
    r"|of course[!,] let'?s"
    r"|absolutely[!,] let'?s"
    r"|i'?m happy to help"
    r"|i'?d be happy to"
    r"|great question[!,])",
    re.IGNORECASE,
)
# Assistant-style service phrases (a tell anywhere in the turn).
_DRIFT_ANY_RE = re.compile(
    r"(let'?s go through (the|this) (steps?|problem) together"
    r"|feel free to ask"
    r"|hope (this|that) helps"
    r"|let me know if you (have any|need))",
    re.IGNORECASE,
)


def looks_like_assistant(text: str) -> bool:
    t = text.strip()
    return bool(_DRIFT_OPEN_RE.match(t) or _DRIFT_ANY_RE.search(t))


class TrainEngagementClassroom(Classroom):
    # Same UserLM view as the validated eval classroom: generic intent in the
    # system slot, the problem delivered via a synthetic student-first opening.
    INTENT = "You are a student trying to solve a math problem."

    def __init__(
        self,
        teacher_model_cfg,
        engagement_model_cfg,
        judge_model_cfg,
        generation_cfg,
        model_save_path,
        log_file_path=None,
        leak_multiplier=0.0,
    ):
        # NOTE: deliberately NOT calling Classroom.__init__ (no student probe /
        # reward model at train time). Only the three models below are loaded.
        self.teacher_model_cfg = teacher_model_cfg
        self.engagement_model_cfg = engagement_model_cfg
        self.judge_model_cfg = judge_model_cfg
        self.generation_cfg = generation_cfg
        self.leak_multiplier = leak_multiplier

        self.teacher_model = ParallelvLLMInference(
            model_path=teacher_model_cfg.model_name_or_path,
            gpu_memory_utilization=teacher_model_cfg.vllm.gpu_memory_utilization,
            gpu_ids=teacher_model_cfg.vllm.gpu_ids,
            max_model_len=teacher_model_cfg.vllm.max_length,
            max_num_seqs=teacher_model_cfg.vllm.max_num_seqs,
            model_save_path=model_save_path,  # -> auto-reload on new checkpoint
            load_and_unload=False,
            enable_sleep_mode=False,
            enforce_eager=teacher_model_cfg.vllm.enforce_eager,
            use_v0=False,
            logging_enabled=log_file_path is not None,
            log_file_path=log_file_path,
        )

        # Overridable so a different simulator (see train_osim_classroom.py)
        # can swap the engine flags, tokenizer and stop tokens without
        # duplicating this whole constructor.
        self._setup_engagement_model(engagement_model_cfg)

        self.judge_model = ParallelvLLMInference(
            model_path=judge_model_cfg.model_name_or_path,
            gpu_memory_utilization=judge_model_cfg.vllm.gpu_memory_utilization,
            gpu_ids=judge_model_cfg.vllm.gpu_ids,
            max_model_len=judge_model_cfg.vllm.max_length,
            max_num_seqs=judge_model_cfg.vllm.max_num_seqs,
            model_save_path=None,
            load_and_unload=False,
            enable_sleep_mode=False,
            enforce_eager=judge_model_cfg.vllm.enforce_eager,
            use_v0=False,
            chat_template_kwargs={"reasoning_effort": "low"},
        )

        # Stop the teacher the moment it starts scripting a fake student
        # line inside its own turn (reward hack found in v1: fabricated
        # "User: ..." exchanges fooled the learning judge while the real
        # student never spoke).
        self.sampling_params_teacher = SamplingParams(
            temperature=teacher_model_cfg.vllm.temperature,
            top_k=teacher_model_cfg.vllm.top_k,
            top_p=teacher_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
            stop=[
                "\nUser:", "\nStudent:", "\nHuman:", "\nLearner:",
                "\nuser:", "\nstudent:",
            ],
        )
        self.sampling_params_engagement = self._build_engagement_sampling_params(
            engagement_model_cfg, generation_cfg
        )
        self.sampling_params_judge = SamplingParams(
            temperature=judge_model_cfg.vllm.temperature,
            top_p=judge_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_judge_attempt,
        )

        self.conversation_sets = []

    # ------------------------------------------------------------------
    # UserLM student turns (same construction as the eval classroom)
    # ------------------------------------------------------------------

    def _setup_engagement_model(self, engagement_model_cfg):
        """UserLM-8b: its chat template omits the BOS the Llama base expects,
        so userlm_mode renders the template and prepends BOS manually."""
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
            userlm_mode=True,  # render template + BOS, use .generate()
        )
        self.engagement_tokenizer = AutoTokenizer.from_pretrained(
            engagement_model_cfg.model_name_or_path
        )
        self.eot_id = self.engagement_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.endconv_id = self.engagement_tokenizer.convert_tokens_to_ids(
            END_CONV_TOKEN
        )

    def _build_engagement_sampling_params(self, engagement_model_cfg, generation_cfg):
        """UserLM per its model card; both <|eot_id|> and <|endconversation|>
        must stop generation or the model drifts into assistant-style text."""
        return SamplingParams(
            temperature=engagement_model_cfg.vllm.temperature,
            top_p=engagement_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
            stop_token_ids=[self.eot_id, self.endconv_id],
        )

    def _opening_turn(self, conv: Conversation) -> str:
        return (
            f"Here's a problem I'm trying to solve:\n\n{conv.problem}"
            "\n\nCan you help me?"
        )

    def _build_userlm_messages(self, conv: Conversation):
        messages = [
            {"role": "system", "content": self.INTENT},
            {"role": "user", "content": self._opening_turn(conv)},
        ]
        for m in conv.conversation:
            role = "user" if m["role"] == "student" else "assistant"
            messages.append({"role": role, "content": _clean_for_userlm(m["content"])})
        return messages

    # Max resamples of a student turn that reads as assistant-drift before we
    # declare the drift persistent and truncate the dialogue.
    MAX_DRIFT_RESAMPLES = 2

    def _generate_student_turns(self, conversations):
        # Rejection-sample UserLM turns that drift into assistant persona:
        # drift is a bad stochastic sample (temp 1.0), so resampling usually
        # repairs it. Persistent drift is treated as a simulator failure: the
        # dialogue is truncated at that point and the drifted turn is NOT
        # added to the transcript (no reward can be earned from or after it).
        pending = list(conversations)
        for attempt in range(self.MAX_DRIFT_RESAMPLES + 1):
            if not pending:
                break
            prompts = [self._build_userlm_messages(c) for c in pending]
            responses = self.engagement_model.run_batch(
                prompts, self.sampling_params_engagement
            )
            retry = []
            for conv, response in zip(pending, responses):
                out = response.outputs[0]
                raw = out.text
                clean = raw.replace(END_CONV_TOKEN, "").strip()
                explicit_end = (
                    out.stop_reason == self.endconv_id
                    or self.endconv_id in (out.token_ids or [])
                    or END_CONV_TOKEN in raw
                )
                # Blank turn == silent disengagement (student checked out).
                ended = explicit_end or clean == ""
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
                    conv.disengage_reason = "explicit" if explicit_end else "silence"
                    conv.state = ConversationState.END
            pending = retry

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def sample_conversations(self, problems, answers, meta: dict = {},
                             solutions=None):
        # solutions is positionally aligned with problems; None disables the
        # reference-solution block in the teacher prompt.
        sols = solutions if solutions is not None else [None] * len(problems)
        conversations = []
        for problem, answer, sol in zip(problems, answers, sols):
            conv = Conversation(
                problem, answer, self.generation_cfg,
                forced_type=ConversationType.GUIDED,
                reference_solution=sol,
            )
            conv.disengaged = False
            conv.disengage_reason = None
            conv.role_drifted = False
            conv.drift_resamples = 0
            conversations.append(conv)
        for conv in conversations:
            conv.start_conversation()

        round_counter = 1
        while any(
            c.state in (ConversationState.TEACHER_TURN, ConversationState.STUDENT_TURN)
            for c in conversations
        ):
            for state in (
                ConversationState.TEACHER_TURN,
                ConversationState.STUDENT_TURN,
            ):
                todo = [c for c in conversations if c.state == state]
                if not todo:
                    continue
                who = "Teacher" if state == ConversationState.TEACHER_TURN else "Student"
                logger.info(f"===== Turn {round_counter}: {who} ({len(todo)} convs) =====")
                start = time.time()
                if state == ConversationState.TEACHER_TURN:
                    prompts = [c.get_conversation() for c in todo]
                    responses = self.teacher_model.run_batch(
                        prompts, self.sampling_params_teacher, meta
                    )
                    for c, r in zip(todo, responses):
                        c.add_message(r.outputs[0].text)
                else:
                    self._generate_student_turns(todo)
                round_counter += 1
                logger.info(f"Took {time.time() - start:.1f}s")

        # Judge scoring (engagement per student turn + terminal learning).
        logger.info("===== Judge scoring =====")
        start = time.time()
        self._score_conversations(conversations)
        logger.info(f"Judge scoring took {time.time() - start:.1f}s")

        for conv in conversations:
            conv.state = ConversationState.END

        gc.collect()
        torch.cuda.empty_cache()
        self.conversation_sets.append(conversations)
        return conversations

    # ------------------------------------------------------------------
    # Judge scoring
    # ------------------------------------------------------------------

    def _judge_batch_with_retry(self, messages, extractor, defaults):
        """Run judge on messages; one retry for unparsable outputs; then
        fall back to `defaults` (logged)."""
        if not messages:
            return []
        responses = self.judge_model.run_batch(messages, self.sampling_params_judge)
        parsed = [extractor(r.outputs[0].text) for r in responses]
        fail_idx = [i for i, p in enumerate(parsed) if p is None]
        if fail_idx:
            logger.info(f"Judge: retrying {len(fail_idx)} unparsable outputs")
            retry = self.judge_model.run_batch(
                [messages[i] for i in fail_idx], self.sampling_params_judge
            )
            for i, r in zip(fail_idx, retry):
                parsed[i] = extractor(r.outputs[0].text)
        n_default = sum(1 for p in parsed if p is None)
        if n_default:
            logger.warning(f"Judge: {n_default}/{len(parsed)} defaulted after retry")
        return [p if p is not None else dict(defaults) for p in parsed]

    def _score_conversations(self, conversations):
        # ---- per-turn engagement (+ learning evidence) ----
        eng_messages, slots = [], []  # slots: (conv, index into conv.turn_scores)
        for conv in conversations:
            turns = conv._get_hidden_conversation()
            conv.turn_scores = []
            for i, m in enumerate(turns):
                if m["role"] != "student":
                    continue
                if m["content"].strip() == "":
                    # Blank turn: all-zero by rubric, no judge call needed.
                    conv.turn_scores.append(dict(ZERO_ENGAGEMENT_SCORES))
                    continue
                eng_messages.append(
                    [
                        {"role": "system", "content": ENGAGEMENT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_engagement_user_prompt(
                                turns[:i], m["content"]
                            ),
                        },
                    ]
                )
                slots.append((conv, len(conv.turn_scores)))
                conv.turn_scores.append(None)

        results = self._judge_batch_with_retry(
            eng_messages, extract_engagement_scores, DEFAULT_ENGAGEMENT_SCORES
        )
        for (conv, idx), scores in zip(slots, results):
            conv.turn_scores[idx] = scores

        # ---- terminal learning outcome ----
        learn_messages = [
            [
                {"role": "system", "content": LEARNING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_learning_user_prompt(
                        conv.problem, conv.answer, conv._get_hidden_conversation()
                    ),
                },
            ]
            for conv in conversations
        ]
        results = self._judge_batch_with_retry(
            learn_messages, extract_learning_scores, DEFAULT_LEARNING_SCORES
        )
        for conv, scores in zip(conversations, results):
            conv.learning_scores = scores

    # ------------------------------------------------------------------
    # Rewards (called by the server endpoints)
    # ------------------------------------------------------------------

    def get_engagement_reward(self, conversation: Conversation) -> float:
        # max_turns counts TOTAL messages (both roles) and is only checked
        # after a teacher turn, so a teacher-first GUIDED dialogue affords
        # max_turns//2 student turns -- that, not max_turns, is the capacity
        # the engagement sum is normalized against.
        return engagement_scalar(
            getattr(conversation, "turn_scores", []),
            max_student_turns=self.generation_cfg.max_turns // 2,
        )

    def get_learning_reward(self, conversation: Conversation) -> float:
        scores = getattr(conversation, "learning_scores", None)
        raw = learning_scalar(scores)
        # Leak gate: a tutor that gives the solution away forfeits learning
        # credit (multiplier configurable via reward.leak_multiplier).
        if scores is not None and scores.get("tutor_leaked", False):
            raw *= self.leak_multiplier
        # Participation gate (discrete stand-in for the paper's survival
        # factor): learning only counts if the real student took part.
        # 0 non-empty student turns -> 0, 1 -> half credit, >=2 -> full.
        n_real = sum(
            1
            for m in conversation.conversation
            if m["role"] == "student" and m["content"].strip()
        )
        return raw * min(1.0, n_real / 2.0)
