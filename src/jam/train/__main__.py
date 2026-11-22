"""python -m jam.train --config CONFIG --batches DIR --output DIR [--backbone DIR] [--init-from CHECKPOINT] [key=value ...]"""

import argparse

from ..config import load_config
from .loop import train


def main():
    parser = argparse.ArgumentParser(description="Train JAM on prepared tensor batches.")
    parser.add_argument("--config", required=True, nargs="+")
    parser.add_argument("--batches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone", help="local video-backbone directory with a transformer/ subfolder")
    parser.add_argument("--init-from", help="release checkpoint used for a warm start")
    parser.add_argument("overrides", nargs="*", help="dotted key=value configuration overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.backbone:
        cfg.model.backbone = args.backbone
    train(cfg, args.batches, args.output, init_from=args.init_from)


if __name__ == "__main__":
    main()
