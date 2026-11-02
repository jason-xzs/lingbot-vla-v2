"""Evaluate a RoboTwin checkpoint with clean/seen and randomized/unseen."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
POLICY = CODE / 'robotwin_policy/linbot-vla-v2'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robotwin-root', type=Path, default=os.environ.get('ROBOTWIN_ROOT'))
    p.add_argument('--infer-python', type=Path, default=os.environ.get('LINGBOT_PYTHON'))
    p.add_argument('--sim-python', type=Path, default=os.environ.get('ROBOTWIN_PYTHON'))
    p.add_argument('--checkpoint', type=Path, default=os.environ.get('LINGBOT_CHECKPOINT'))
    p.add_argument('--setting', choices=('clean', 'random', 'both'), default='both')
    p.add_argument('--task', default='all')
    p.add_argument('--test-num', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', default='0')
    p.add_argument('--port', type=int, default=8006)
    p.add_argument('--log-dir', type=Path, default=CODE / 'runs/eval')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.checkpoint is None:
        p.error('Set --checkpoint or LINGBOT_CHECKPOINT to the model to evaluate')
    if args.robotwin_root is None or args.infer_python is None or args.sim_python is None:
        p.error('Set --robotwin-root, --infer-python and --sim-python (or corresponding documented environment variables)')
    robotwin = args.robotwin_root.expanduser().resolve()
    if not (robotwin / 'script/eval_policy.py').is_file():
        p.error(f'Not a RoboTwin checkout: {robotwin}')
    if args.test_num <= 0 or args.seed < 0:
        p.error('test-num must be positive and seed non-negative')
    for exe in (args.infer_python, args.sim_python):
        if not exe.is_file():
            p.error(f'Python interpreter missing: {exe}')
    # Use our local runner, passing RoboTwin explicitly; no overwriting an existing policy.
    if not args.dry_run:
        target = robotwin / 'policy/linbot-vla-v2'
        if target.exists():
            for name in ('__init__.py', 'deploy_policy.py', 'msgpack_numpy.py'):
                if not (target / name).is_file() or (target / name).read_bytes() != (POLICY / name).read_bytes():
                    raise FileExistsError(f'Different existing policy at {target}; use a separate RoboTwin checkout')
        else:
            shutil.copytree(POLICY, target, ignore=shutil.ignore_patterns('__pycache__'))
    env = os.environ.copy()
    env.update(QWEN3VL_PATH=os.environ.get('QWEN3VL_PATH', str(CODE / 'pretrained/Qwen3-VL-4B-Instruct')),
               ROBOTWIN_NORM_STATS=str(CODE / 'assets/norm_stats/robotwin_clean.json'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    label = f'robotwin_seed{args.seed}_n{args.test_num}'
    cmd = [sys.executable, str(POLICY / 'run_eval.py'), '--robotwin-root', str(robotwin),
           '--lingbot-root', str(CODE), '--checkpoint', str(args.checkpoint.expanduser().resolve()),
           '--infer-python', str(args.infer_python.resolve()), '--sim-python', str(args.sim_python.resolve()),
           '--task', args.task, '--setting', args.setting, '--instruction-type', 'auto',
           '--test-num', str(args.test_num), '--seed', str(args.seed), '--inference-seed', '42',
           '--use-length', '50', '--precision', 'fp32', '--gpu', args.gpu, '--port', str(args.port),
           '--ckpt-setting', label, '--log-dir', str(args.log_dir.resolve())]
    if args.dry_run:
        cmd.append('--dry-run')
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=robotwin, env=env, check=True)


if __name__ == '__main__':
    main()
