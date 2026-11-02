"""Start a LingBot server and evaluate RoboTwin clean/random tasks."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


POLICY_DIR = Path(__file__).resolve().parent
LINGBOT_ROOT = POLICY_DIR.parents[1]
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", str(LINGBOT_ROOT.parent / "RoboTwin")))
CHECKPOINT = Path(os.environ.get("LINGBOT_CHECKPOINT", str(LINGBOT_ROOT / "runs/merged/checkpoints/wise_50k60k_a0.75/hf_ckpt")))
SETTINGS = {"clean": ["demo_clean"], "demo_clean": ["demo_clean"],
            "random": ["demo_randomized"], "randomized": ["demo_randomized"],
            "demo_randomized": ["demo_randomized"], "both": ["demo_clean", "demo_randomized"]}


def parse_args():
    global ROBOTWIN_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, default=ROBOTWIN_ROOT)
    parser.add_argument("--task", default="all", help="Task name, comma-separated names, or all")
    parser.add_argument("--tasks-file", type=Path, default=POLICY_DIR / "tasks.txt")
    parser.add_argument("--setting", choices=SETTINGS, default="both")
    parser.add_argument("--gpu", default="0", help="Physical GPU index or UUID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--ckpt-setting", help="Result directory label; inferred from checkpoint by default")
    parser.add_argument("--lingbot-root", type=Path, default=LINGBOT_ROOT)
    parser.add_argument("--infer-python", type=Path, default=os.environ.get(
        "LINGBOT_PYTHON", sys.executable))
    parser.add_argument("--sim-python", type=Path, default=os.environ.get(
        "ROBOTWIN_PYTHON", sys.executable))
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="RoboTwin episode seed")
    parser.add_argument("--inference-seed", type=int, default=42)
    parser.add_argument("--instruction-type", choices=("auto", "seen", "unseen"), default="auto",
                        help="auto uses seen for clean and unseen for randomized")
    parser.add_argument("--use-length", type=int, default=50)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--task-timeout", type=float, default=0, help="Seconds per task; 0 disables the limit")
    parser.add_argument("--log-dir", type=Path, help="Parent directory for a new run directory")
    parser.add_argument("--external-server", action="store_true", help="Use an already-running serve.py instance")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ROBOTWIN_ROOT = args.robotwin_root.expanduser().resolve()
    if args.test_num <= 0 or args.startup_timeout <= 0 or args.task_timeout < 0:
        parser.error("test-num/startup-timeout must be positive and task-timeout non-negative")
    if args.seed < 0 or not 1 <= args.use_length <= 50 or not 1 <= args.port <= 65535:
        parser.error("seed must be non-negative, use-length 1..50, port 1..65535")
    for key in ("checkpoint", "lingbot_root", "infer_python", "sim_python", "tasks_file"):
        setattr(args, key, getattr(args, key).expanduser().absolute())
    if (args.checkpoint / "hf_ckpt").is_dir():
        args.checkpoint = args.checkpoint / "hf_ckpt"
    if args.ckpt_setting is None:
        step = args.checkpoint.parent.name if args.checkpoint.name == "hf_ckpt" else args.checkpoint.name
        args.ckpt_setting = "robotwin_clean_step50000" if step == "global_step_50000" else step
    if Path(args.ckpt_setting).name != args.ckpt_setting or args.ckpt_setting in (".", ".."):
        parser.error("ckpt-setting must be a directory label, without slashes")
    if args.task == "all":
        tasks = [line.split("#", 1)[0].strip() for line in args.tasks_file.read_text().splitlines()]
        tasks = [task for task in tasks if task]
    else:
        tasks = [task.strip() for task in args.task.split(",") if task.strip()]
    if not tasks or len(tasks) != len(set(tasks)):
        parser.error("Task list must be non-empty and contain no duplicates")
    for task in tasks:
        if not task.isidentifier() or not (ROBOTWIN_ROOT / "envs" / f"{task}.py").is_file():
            parser.error(f"Unknown RoboTwin task: {task}")
    for config in SETTINGS[args.setting]:
        if not (ROBOTWIN_ROOT / "task_config" / f"{config}.yml").is_file():
            parser.error(f"Missing task config: {config}")
    return args, tasks


def child_env(python, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTHONUNBUFFERED="1", CONDA_PREFIX=str(python.parent.parent))
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")
    return env


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    # Every child starts its own session; never signal the caller's process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def wait_for_server(args, server):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"Inference server exited with {server.returncode}; see inference.log")
        try:
            with opener.open(f"http://{args.host}:{args.port}/healthz", timeout=2) as response:
                if response.status == 200 and response.read().strip() == b"OK":
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise TimeoutError("Inference server startup timed out; see inference.log")


def commands(args, tasks):
    server = [str(args.infer_python), "-B", str(POLICY_DIR / "serve.py"),
              "--checkpoint", str(args.checkpoint), "--lingbot-root", str(args.lingbot_root),
              "--host", args.host, "--port", str(args.port), "--use-length", str(args.use_length),
              "--precision", args.precision, "--seed", str(args.inference_seed)]
    if args.compile:
        server.append("--compile")
    jobs = []
    for setting in SETTINGS[args.setting]:
        instruction_type = args.instruction_type
        if instruction_type == "auto":
            instruction_type = "seen" if setting == "demo_clean" else "unseen"
        for task in tasks:
            command = [str(args.sim_python), "-B", "script/eval_policy.py", "--config",
                       str(POLICY_DIR / "deploy_policy.yml"), "--overrides"]
            overrides = {"task_name": task, "task_config": setting, "policy_name": "linbot-vla-v2",
                         "ckpt_setting": args.ckpt_setting, "seed": args.seed,
                         "test_num": args.test_num, "instruction_type": instruction_type,
                         "host": args.host, "port": args.port,
                         "expected_checkpoint": str(args.checkpoint.resolve()),
                         "expected_precision": args.precision, "expected_use_length": args.use_length,
                         "expected_compile": args.compile, "expected_seed": args.inference_seed}
            for key, value in overrides.items():
                command.extend(["--" + key, str(value)])
            jobs.append((setting, task, command))
    return server, jobs


def write_summary(run_dir, rows):
    temporary_json = run_dir / "summary.json.tmp"
    temporary_json.write_text(json.dumps(rows, indent=2) + "\n")
    temporary_json.replace(run_dir / "summary.json")
    temporary_csv = run_dir / "summary.csv.tmp"
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["setting", "task", "status", "returncode",
                                                    "success_rate", "elapsed_seconds", "result", "log"])
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(run_dir / "summary.csv")


def main():
    args, tasks = parse_args()
    server_cmd, jobs = commands(args, tasks)
    if args.dry_run:
        print(f"cwd: {ROBOTWIN_ROOT}\nCUDA_VISIBLE_DEVICES={args.gpu}")
        if not args.external_server:
            print("Server:", shlex.join(server_cmd))
        for setting, task, command in jobs:
            print(f"{setting}/{task}: {shlex.join(command)}")
        print(f"Total: {len(jobs)} task/scenario pairs, {len(jobs) * args.test_num} valid episodes")
        return 0

    interpreters = [args.sim_python] if args.external_server else [args.sim_python, args.infer_python]
    for python in interpreters:
        if not python.is_file():
            raise FileNotFoundError(f"Python interpreter not found: {python}")
    if not args.external_server:
        # Refuse occupied ports so readiness cannot be satisfied by another server.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((args.host, args.port))
    log_root = args.log_dir or POLICY_DIR / "logs"
    run_dir = log_root.expanduser().absolute() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    print(f"Run directory: {run_dir}", flush=True)
    server = current = None
    rows = []
    def on_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_signal)
    try:
        if not args.external_server:
            with (run_dir / "inference.log").open("w") as log:
                server = subprocess.Popen(server_cmd, cwd=ROBOTWIN_ROOT,
                                          env=child_env(args.infer_python, args.gpu),
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print("Waiting for inference server...", flush=True)
        wait_for_server(args, server)
        print("Inference server ready", flush=True)
        for index, (setting, task, command) in enumerate(jobs, 1):
            result_dir = ROBOTWIN_ROOT / "eval_result" / task / "linbot-vla-v2" / setting / args.ckpt_setting
            previous = {p: p.stat().st_mtime_ns for p in result_dir.glob("*/_result.txt")}
            log_path = run_dir / f"{setting}__{task}.log"
            print(f"[{index}/{len(jobs)}] {setting}/{task}: {log_path}", flush=True)
            started = time.monotonic()
            timed_out = False
            with log_path.open("w") as log:
                current = subprocess.Popen(command, cwd=ROBOTWIN_ROOT,
                                           env=child_env(args.sim_python, args.gpu),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while current.poll() is None:
                    if server is not None and server.poll() is not None:
                        raise RuntimeError("Inference server exited during evaluation; see inference.log")
                    if args.task_timeout and time.monotonic() - started > args.task_timeout:
                        timed_out = True
                        stop_process(current)
                        break
                    time.sleep(0.5)
            results = [p for p in result_dir.glob("*/_result.txt")
                       if p.stat().st_mtime_ns != previous.get(p)]
            result = max(results, key=lambda p: p.stat().st_mtime_ns) if results else None
            score = None
            if result:
                try:
                    score = float(result.read_text().strip().splitlines()[-1])
                    if not 0 <= score <= 1:
                        score = None
                except (ValueError, IndexError):
                    pass
            status = "ok" if current.returncode == 0 and score is not None else "failed"
            if timed_out:
                status = "timeout"
            rows.append({"setting": setting, "task": task, "status": status,
                         "returncode": current.returncode, "success_rate": score,
                         "elapsed_seconds": round(time.monotonic() - started, 2),
                         "result": str(result) if result else "", "log": str(log_path)})
            write_summary(run_dir, rows)
            print(f"  {status}; success_rate={score}; {rows[-1]['elapsed_seconds']}s", flush=True)
            current = None
        print(f"Summary: {run_dir / 'summary.csv'}", flush=True)
        return 0 if all(row["status"] == "ok" for row in rows) else 1
    except KeyboardInterrupt:
        print("Interrupted; stopping owned processes.", file=sys.stderr)
        return 130
    finally:
        stop_process(current)
        stop_process(server)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
