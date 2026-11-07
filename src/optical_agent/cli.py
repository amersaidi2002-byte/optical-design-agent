"""Reproducible CPU workflow, exposed as optical-agent."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("generate", help="Simulate 10,000 designs and split families")
    data.add_argument("--output", default="data/designs.npz")
    data.add_argument("--families", type=int, default=250)
    data.add_argument("--members", type=int, default=40)
    data.add_argument("--seed", type=int, default=2026)
    fit = sub.add_parser(
        "train", help="Train conditional VAE using training/validation families"
    )
    fit.add_argument("--data", default="data/designs.npz")
    fit.add_argument("--output", default="models/cvae.pt")
    fit.add_argument("--epochs", type=int, default=250)
    fit.add_argument("--seed", type=int, default=2026)
    bench = sub.add_parser(
        "benchmark", help="24 held-out tasks × 3 paired search seeds"
    )
    bench.add_argument("--data", default="data/designs.npz")
    bench.add_argument("--model", default="models/cvae.pt")
    bench.add_argument("--output", default="reports/benchmark.json")
    bench.add_argument("--budget", type=int, default=192)
    bench.add_argument("--tasks", type=int, default=24)
    plot = sub.add_parser(
        "report", help="Render plots and Markdown from saved evidence"
    )
    plot.add_argument("--benchmark", default="reports/benchmark.json")
    plot.add_argument("--training", default="models/cvae.json")
    plot.add_argument("--output", default="reports/figures")
    demo = sub.add_parser("design", help="Propose and refine an optical prescription")
    demo.add_argument("--focal", type=float, default=50)
    demo.add_argument("--half-fov", type=float, default=5)
    demo.add_argument("--f-number", type=float, default=6)
    demo.add_argument("--budget", type=int, default=192)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--model", default="models/cvae.pt")
    demo.add_argument("--output", default="reports/demo.json")
    args = parser.parse_args()
    if args.command == "generate":
        from .dataset import generate

        result = generate(args.output, args.families, args.members, args.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        from .model import train

        result = train(args.data, args.output, args.epochs, args.seed)
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    elif args.command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(args.data, args.model, args.output, args.budget, args.tasks)
        print(json.dumps(result["summary"], indent=2))
    elif args.command == "report":
        from .plots import make_report

        make_report(args.benchmark, args.training, args.output)
    else:
        import torch

        from .model import load_model
        from .plots import design_plot
        from .search import optimize

        torch.set_num_threads(2)
        result = optimize(
            [args.focal, args.half_fov, args.f_number],
            args.budget,
            args.seed,
            load_model(args.model),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        design_plot(result, output.with_suffix(".png"))
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
