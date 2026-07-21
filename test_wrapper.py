"""Isolate the one porting risk: does their ParallelvLLMInference (written for
vllm 0.8.3) work on vllm 0.23? Mirrors how Classroom uses it: construct ->
sleep() -> run_batch(chat msgs) -> cleanup.
Run from PedagogicalRL/ with env_eval.sh sourced, CUDA_VISIBLE_DEVICES=0.
"""
from src.vllm.data_parallel_vllm import ParallelvLLMInference
from vllm import SamplingParams


def main():
    m = ParallelvLLMInference(
        model_path="eth-nlped/TutorRL-7B",
        gpus_per_instance=1,
        gpu_memory_utilization=0.45,
        max_model_len=4096,
        max_num_seqs=8,
        max_number_of_instances=1,
        enforce_eager=True,
        enable_sleep_mode=True,
        load_and_unload=True,
    )
    print("[test] constructed OK", flush=True)
    m.sleep()
    print("[test] sleep OK", flush=True)
    out = m.run_batch(
        [[{"role": "user", "content": "What is 2+2? Answer in one short sentence."}]],
        SamplingParams(temperature=0.6, max_tokens=40),
    )
    print("[test] run_batch OK", flush=True)
    print("[test] OUTPUT:", repr(out[0].outputs[0].text), flush=True)
    m.cleanup()
    print("[test] DONE", flush=True)


if __name__ == "__main__":
    main()
