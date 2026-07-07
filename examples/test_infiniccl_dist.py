"""Tensor-parallel inference over the standalone InfiniCCL library.

The default standalone InfiniCCL path uses InfiniLM's normal single-process
execution model: one process hosts every tensor-parallel rank, and each rank
worker thread owns one standalone InfiniCCL communicator initialized through the
native CCL `GetUniqueId` + `CommInitRank` flow.

Launch:

    cd InfiniLM
    export PYTHONPATH=$PWD/python
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/.infini/lib:$LD_LIBRARY_PATH
    export INFINILM_USE_INFINICCL=1
    export INFINICCL_LIB=$HOME/ninetoothed/InfiniCCL/install/lib/libinfiniccl.so
    python examples/test_infiniccl_dist.py --model ~/models/Qwen3-0.6B --tp 2

Set INFINILM_INFINICCL_COMM_MODE=mpi only for the legacy one-process-per-rank
experiment, launched with `mpirun -np <tp>`.
"""

import argparse
import os
import time

RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))


def rank0_print(*args, **kwargs):
    if RANK == 0:
        print(*args, **kwargs, flush=True)


def comm_mode(env=os.environ):
    return env.get("INFINILM_INFINICCL_COMM_MODE", "ccl_single_process")


def validate_launch(tp, env=os.environ):
    if tp <= 1:
        return
    if env.get("INFINILM_USE_INFINICCL") != "1":
        raise SystemExit("tp > 1 requires INFINILM_USE_INFINICCL=1")

    mode = comm_mode(env)
    if mode == "ccl_single_process":
        return
    if mode == "mpi":
        world_size = int(env.get("OMPI_COMM_WORLD_SIZE", "1"))
        if world_size != tp:
            raise SystemExit(
                f"INFINILM_INFINICCL_COMM_MODE=mpi requires `mpirun -np {tp}` "
                f"(current OMPI_COMM_WORLD_SIZE={world_size})"
            )
        return
    raise SystemExit(
        "INFINILM_INFINICCL_COMM_MODE must be `ccl_single_process` or `mpi`"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model directory (HF layout)")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Static KV cache length",
    )
    parser.add_argument(
        "--prompt",
        default="Briefly introduce the Great Wall of China.",
        help="User prompt",
    )
    args = parser.parse_args()

    validate_launch(args.tp)

    from infinilm.llm.llm import LLM

    rank0_print(f"[rank {RANK}] loading {args.model} (tp={args.tp}, device={args.device})")
    t0 = time.time()
    model = LLM(
        model_path=os.path.expanduser(args.model),
        device=args.device,
        tensor_parallel_size=args.tp,
        cache_type="static",
        max_cache_len=args.max_cache_len,
        max_tokens=args.max_new_tokens,
        temperature=1.0,
        top_k=1,  # greedy: keeps all ranks' decisions identical
        top_p=1.0,
    )
    rank0_print(f"model ready in {time.time() - t0:.1f}s")

    conversations = [[{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]]
    t1 = time.time()
    outputs = model.chat(messages=conversations)
    t2 = time.time()

    for output in outputs:
        rank0_print("=== Prompt ===")
        rank0_print(output.prompt)
        rank0_print("=== Response ===")
        rank0_print(output.outputs[0].text)
        n_tokens = len(output.outputs[0].token_ids)
        rank0_print(
            f"\n[{n_tokens} tokens in {t2 - t1:.2f}s, "
            f"{n_tokens / max(t2 - t1, 1e-9):.2f} tok/s, tp={args.tp}, "
            f"comm=standalone InfiniCCL ({comm_mode(os.environ)})]"
        )


if __name__ == "__main__":
    main()
