# ---------------------------------------------------------------
# Convert a Lightning checkpoint (.ckpt) into a plain state dict (.bin), the
# format the README uses for --model.ckpt_path (evaluation / fine-tuning).
#
# Usage:
#   python tools/export_weights.py --ckpt-dir checkpoints/run --out best.bin   # best-*.ckpt, else last.ckpt
#   python tools/export_weights.py --ckpt path/to/file.ckpt --out best.bin
# ---------------------------------------------------------------

import argparse
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", help="Lightning checkpoint file")
    source.add_argument("--ckpt-dir", help="ModelCheckpoint folder: newest best-*.ckpt, else last.ckpt")
    parser.add_argument("--out", required=True, help="output .bin")
    args = parser.parse_args()

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        best = sorted(Path(args.ckpt_dir).glob("best-*.ckpt"), key=os.path.getmtime)
        ckpt_path = best[-1] if best else Path(args.ckpt_dir) / "last.ckpt"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {k.replace("._orig_mod", ""): v for k, v in ckpt["state_dict"].items()}
    torch.save(state_dict, args.out)
    print(f"{ckpt_path} (epoch {ckpt.get('epoch')}) -> {args.out}: {len(state_dict)} tensors")


if __name__ == "__main__":
    main()
