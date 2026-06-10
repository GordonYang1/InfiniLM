"""SPMD driver for tensor-parallel inference over the standalone InfiniCCL library.

Every tensor-parallel rank is a separate MPI process running this same script
(InfiniCCL follows the MPI process model, unlike InfiniCore's built-in infiniccl,
which hosts all ranks in one process). Rank 0 samples; the sampled token ids are
broadcast to the other ranks inside the engine, so all processes stay in lockstep.

Launch (2 ranks sharing one local GPU is fine; the engine wraps device ids):

    cd InfiniLM
    export PYTHONPATH=$PWD/python
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/.infini/lib:$LD_LIBRARY_PATH
    mpirun -np 2 --oversubscribe \
        -x INFINILM_USE_INFINICCL=1 \
        -x INFINICCL_LIB=$HOME/ninetoothed/InfiniCCL/install/lib/libinfiniccl.so \
        -x LD_LIBRARY_PATH -x PYTHONPATH \
        python examples/test_infiniccl_dist.py --model ~/models/Qwen3-0.6B --tp 2

Greedy decoding (top_k=1) keeps every rank's generation identical.
"""

import argparse
import os
import time

RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))


def rank0_print(*args, **kwargs):
    if RANK == 0:
        print(*args, **kwargs, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model directory (HF layout)")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size (= mpirun -np)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Static KV cache length (kept small: 2 ranks share one 6GB GPU locally)",
    )
    parser.add_argument(
        "--prompt",
        default="Briefly introduce the Great Wall of China.",
        help="User prompt",
    )
    args = parser.parse_args()

    if args.tp > 1 and os.environ.get("INFINILM_USE_INFINICCL") != "1":
        raise SystemExit(
            "tp > 1 requires INFINILM_USE_INFINICCL=1 and an mpirun launch; "
            "see the module docstring"
        )

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
            f"comm=standalone InfiniCCL (OMPI)]"
        )


if __name__ == "__main__":
    main()
