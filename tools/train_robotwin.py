"""Train the two-stage RoboTwin clean recipe and merge its checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

CODE = Path(__file__).resolve().parents[1]
SRC = CODE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('stage1', 'stage2', 'merge'), required=True)
    p.add_argument('--base-model', type=Path, default=os.environ.get('LINGBOT_BASE'))
    p.add_argument('--moge-model', type=Path, default=os.environ.get('MOGE_MODEL'))
    p.add_argument('--data-root', type=Path, default=os.environ.get('CLEAN_DATA_ROOT'))
    p.add_argument('--tokenizer', type=Path, default=os.environ.get('QWEN3VL_PATH', str(CODE / 'pretrained/Qwen3-VL-4B-Instruct')))
    p.add_argument('--augmentation-root', type=Path, default=CODE / 'assets/augmentation')
    p.add_argument('--work-dir', type=Path, default=CODE / 'runs')
    p.add_argument('--stage1-checkpoint', type=Path)
    p.add_argument('--stage2-checkpoint', type=Path)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--resume-dataloader', action='store_true')
    p.add_argument('--nproc', type=int, default=6)
    p.add_argument('--master-port', type=int, default=62531)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    work = args.work_dir.expanduser().resolve()
    first = (args.stage1_checkpoint or work / 'stage1/checkpoints/global_step_50000').resolve()
    second = (args.stage2_checkpoint or work / 'stage2/checkpoints/global_step_60000').resolve()
    if args.stage == 'merge':
        cmd = [sys.executable, str(CODE / 'tools/merge_checkpoints.py'), '--a', str(first), '--b', str(second),
               '--alpha', '0.75', '--output', str(work / 'merged/checkpoints/wise_50k60k_a0.75/hf_ckpt')]
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        return
    if args.base_model is None or args.data_root is None:
        p.error('Training requires --base-model (official lingbot-vla-v2-6b) and --data-root (50-task clean LeRobot root)')
    base = args.base_model.expanduser().resolve()
    data = args.data_root.expanduser().resolve()
    if args.nproc < 1 or 384 % (32 * args.nproc):
        p.error('nproc must divide 12 to keep batch_size=384 with micro_batch_size=32')
    tasks = [s.split('#', 1)[0].strip() for s in (CODE / 'robotwin_policy/linbot-vla-v2/tasks.txt').read_text().splitlines()]
    tasks = [s for s in tasks if s]
    assert len(tasks) == len(set(tasks)) == 50
    manifest = []
    for task in tasks:
        root = data / task
        d = json.loads((root / 'meta/robotwin_source.json').read_text())
        assert d['source']['repo_id'] == 'TianxingChen/RoboTwin2.0'
        assert d['source']['path'] == f'dataset/{task}/aloha-agilex_clean_50.zip'
        assert d['instruction_split'] == 'seen' and d['episodes'] == 50
        assert set(d['source_episode_ids']) == set(range(50))
        manifest.append(f'robotwin {root}')
    cfg = yaml.safe_load((CODE / f'configs/vla/robotwin/clean_scene_{args.stage}.yaml').read_text())
    tokenizer = args.tokenizer.expanduser().resolve()
    if not (tokenizer / 'config.json').is_file():
        raise FileNotFoundError(f'Qwen3-VL tokenizer/config missing: {tokenizer}')
    cfg['model'].update(model_path=str(base), tokenizer_path=str(tokenizer))
    output = work / args.stage
    cfg['data'].update(train_path=str(output / 'clean_tasks.txt'),
                       norm_stats_file=str(SRC / 'assets/norm_stats/robotwin_clean.json'),
                       robot_config_root=str(SRC / 'configs/robot_configs'))
    cfg['train'].update(output_dir=str(output), enable_resume=False,
                        gradient_accumulation_steps=384 // (32 * args.nproc),
                        resume_dataloader_state=args.resume_dataloader)
    # The upstream CLI converts YAML null into the literal string "null" for
    # Optional[str]. Omit this field to use its actual None default.
    cfg['train'].pop('load_checkpoint_path', None)
    if args.stage == 'stage2':
        cfg['train']['load_checkpoint_path'] = str(first)
        augmentation = args.augmentation_root.expanduser().resolve()
        library = augmentation / 'clutter_v1/library.npz'
        instructions = augmentation / 'full50_v1/seen_pools.json'
        for path in (library, instructions):
            if not path.is_file():
                raise FileNotFoundError(f'Prepare augmentation assets first: {path}')
        cfg['data']['mixed_augmentation']['scene']['library'] = str(library)
        cfg['data']['instruction_sampling']['pool_manifest'] = str(instructions)
    if args.resume_from:
        cfg['train']['load_checkpoint_path'] = str(args.resume_from.resolve())
    align = cfg['train']['align_params']
    moge = (args.moge_model or base.parent / 'moge-2-vitb-normal/model.pt').resolve()
    align['depth'].update(moge_path=str(moge), morgbd_path=str(base / 'depth/model.pt'))
    align['video'].update(ckpt_path=str(base / 'dino_video/teacher_step_10000.pth'),
                          config_path=str(base / 'dino_video/config.yaml'))
    for f in (base / 'config.json', moge, base / 'depth/model.pt', base / 'dino_video/teacher_step_10000.pth'):
        if not f.is_file():
            raise FileNotFoundError(f)
    if cfg['train'].get('load_checkpoint_path'):
        ckpt = Path(cfg['train']['load_checkpoint_path'])
        if not (ckpt / 'model/.metadata').is_file() or not (ckpt / 'optimizer/.metadata').is_file():
            raise FileNotFoundError(f'DCP resume checkpoint required, including optimizer/scheduler: {ckpt}')
    config_path = output / 'launch_config.yaml'
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', f'--nproc-per-node={args.nproc}',
           '--node-rank=0', '--master-addr=127.0.0.1', f'--master-port={args.master_port}',
           str(SRC / 'tasks/vla/train_lingbotvla.py'), str(config_path)]
    print(f'50 tasks / 2500 clean episodes; stage={args.stage}; max_steps={cfg["train"]["max_steps"]}', flush=True)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        print('DRY RUN: no training or output files created')
        return
    if output.exists() and (output / 'checkpoints').exists() and not args.resume_from:
        raise FileExistsError(f'Use a new --work-dir or an explicit --resume-from: {output}')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'clean_tasks.txt').write_text('\n'.join(manifest) + '\n')
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC) + os.pathsep + env.get('PYTHONPATH', '')
    env.update(TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1',
               SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0')
    import triton
    assembler = Path(triton.__file__).parent / 'backends/nvidia/bin/ptxas'
    if assembler.is_file():
        env['TRITON_PTXAS_PATH'] = str(assembler)
    subprocess.run(cmd, cwd=SRC, env=env, check=True)


if __name__ == '__main__':
    main()
