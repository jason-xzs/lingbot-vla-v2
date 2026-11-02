"""Create one standalone checkpoint by FP32 interpolation, refusing an existing output."""
import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=0.75)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert 0 <= args.alpha <= 1
    roots = [x / 'hf_ckpt' if (x / 'hf_ckpt').is_dir() else x for x in (args.a, args.b)]
    indices = [json.loads((r / 'model.safetensors.index.json').read_text()) for r in roots]
    assert indices[0]['weight_map'] == indices[1]['weight_map'], 'Input sharding/key layout must match'
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    for filename in sorted(set(indices[0]['weight_map'].values())):
        a, b = [load_file(str(r / filename)) for r in roots]
        assert set(a) == set(b)
        for key in a:
            assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
            if a[key].is_floating_point():
                a[key] = ((1 - args.alpha) * a[key].float() + args.alpha * b[key].float()).to(a[key].dtype)
                assert torch.isfinite(a[key]).all()
            else:
                assert torch.equal(a[key], b[key])
        save_file(a, str(args.output / filename), metadata={'format': 'pt'})
        del a, b
        print(filename, flush=True)
    for f in roots[1].iterdir():
        if f.is_file() and f.suffix != '.safetensors' and f.name != 'merge_info.json':
            shutil.copy2(f, args.output / f.name)
    (args.output / 'merge_info.json').write_text(json.dumps({'a': str(roots[0].resolve()),
        'b': str(roots[1].resolve()), 'alpha': args.alpha}, indent=2) + '\n')
    # Keep the inference configuration at the layout expected by serve.py.
    source_config = roots[1].parents[2] / 'lingbotvla_cli.yaml'
    if source_config.exists():
        shutil.copy2(source_config, args.output.parents[2] / 'lingbotvla_cli.yaml')


if __name__ == '__main__':
    main()
