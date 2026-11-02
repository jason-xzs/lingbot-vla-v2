"""Serve the original LingBot VLA v2 policy for RoboTwin evaluation."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = DEFAULT_ROOT / "runs/merged/checkpoints/wise_50k60k_a0.75/hf_ckpt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lingbot-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--norm-stats", type=Path, default=os.environ.get("ROBOTWIN_NORM_STATS"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--use-length", type=int, default=50)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--seed", type=int, default=42, help="Inference RNG seed")
    parser.add_argument("--check", action="store_true", help="Validate files without loading model")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if not 1 <= args.use_length <= 50:
        parser.error("use-length must be in 1..50 for this checkpoint")
    return args


def validate_files(args):
    import yaml

    root = args.lingbot_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if (checkpoint / "hf_ckpt").is_dir():
        checkpoint = checkpoint / "hf_ckpt"
    required = [root / "deploy/lingbot_vla_v2_policy.py",
                root / "configs/robot_configs/robotwin.yaml", checkpoint / "config.json",
                checkpoint / "model.safetensors.index.json",
                checkpoint.parents[2] / "lingbotvla_cli.yaml"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required inference file is missing: {path}")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards:
        raise ValueError("Checkpoint index has no weight shards")
    for shard in shards:
        if not (checkpoint / shard).is_file():
            raise FileNotFoundError(checkpoint / shard)
    training = yaml.safe_load(required[-1].read_text())
    norm = args.norm_stats or Path(training["data"]["norm_stats_file"])
    if not norm.is_absolute():
        norm = root / norm
    if not norm.is_file():
        raise FileNotFoundError(f"Training normalization statistics not found: {norm}")
    tokenizer = Path(os.environ.get("QWEN3VL_PATH", training["model"]["tokenizer_path"]))
    if not tokenizer.is_absolute():
        tokenizer = root / tokenizer
    if not (tokenizer / "config.json").is_file():
        raise FileNotFoundError(f"Local Qwen3-VL config/tokenizer not found: {tokenizer}")
    return root, checkpoint, norm.resolve(), tokenizer.resolve()


def main():
    args = parse_args()
    root, checkpoint, norm, tokenizer = validate_files(args)
    metadata = {"protocol": "lingbot-vla-v2/robotwin-v1", "checkpoint": str(checkpoint),
                "norm_stats": str(norm), "use_length": args.use_length, "action_dim": 14,
                "precision": args.precision, "compile": args.compile, "seed": args.seed}
    print(json.dumps(metadata, indent=2), flush=True)
    if args.check:
        print("PASS: checkpoint, training config, tokenizer and normalization files exist")
        return

    # Official reset() resolves robot config relative to this working directory.
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["QWEN3VL_PATH"] = str(tokenizer)
    os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")
    # Match the training launcher: the host exports CUDA 13 ptxas, while this
    # model environment's Triton needs its bundled CUDA 12.8 assembler.
    triton_spec = importlib.util.find_spec("triton")
    if triton_spec is not None and triton_spec.origin:
        bundled_ptxas = Path(triton_spec.origin).parent / "backends/nvidia/bin/ptxas"
        if bundled_ptxas.is_file():
            os.environ["TRITON_PTXAS_PATH"] = str(bundled_ptxas)
            print(f"Triton ptxas: {bundled_ptxas}", flush=True)
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, set_seed_everywhere
    from deploy.websocket_policy_server import WebsocketPolicyServer

    set_seed_everywhere(args.seed)
    model = LingbotVLAv2Server(
        path_to_pi_model=str(checkpoint), robot_norm_path=str(norm),
        use_length=args.use_length, chunk_ret=True,
        use_bf16=args.precision == "bf16", use_fp32=args.precision == "fp32",
        use_compile=args.compile,
    )
    model.reset("robotwin")
    WebsocketPolicyServer(model, host=args.host, port=args.port,
                          metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
