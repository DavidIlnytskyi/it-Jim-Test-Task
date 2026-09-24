#!/usr/bin/env python3
"""Generate README figures from saved ResNet-50 experiments.

Install dependencies:
    python -m pip install matplotlib numpy pillow torch torchvision

Generate all four figures (CPU by default; no model download or training):
    python generate_visualizations.py

Generate just the two figures backed by saved JSON metrics:
    python generate_visualizations.py --metrics-only

Optional: --device cuda --batch-size 32 --output-dir assets/results
Paths default to this script's directory, regardless of the working directory.
Charts are exported as SVG and PNG; the prediction gallery is exported as PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "baseline": "Baseline",
    "data-augmentation": "Data augmentation",
    "crop-faces": "Face cropping",
    "weighted-loss": "Weighted loss",
    "focal-loss": "Focal loss",
    "balance-sampling": "Balanced sampling",
    "oversampling": "Oversampling",
}
CLASS_NAMES = ["Artifact", "Artifact-free"]
BLUE = "#2563EB"
ORANGE = "#EA8C23"


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#334155",
        "text.color": "#172033",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })


def save_figure(fig, output_dir, name, formats=("svg", "png")):
    for extension in formats:
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def read_results(runs_dir):
    results = {}
    for name in EXPERIMENTS:
        path = runs_dir / name / "training_results.json"
        data = json.loads(path.read_text())
        for key in ("train_loss", "valid_loss", "metrics"):
            values = np.asarray(data[key], dtype=float)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError(f"{path}: {key} must contain finite values")
        if len({len(data[k]) for k in ("train_loss", "valid_loss", "metrics")}) != 1:
            raise ValueError(f"{path}: epoch histories must have matching lengths")
        scores = np.asarray([*data["metrics"], data["test_micro_f1"]])
        if not (np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()):
            raise ValueError(f"{path}: F1 scores must be in [0, 1]")
        results[name] = data
    return results


def plot_comparison(results, output_dir):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(EXPERIMENTS))
    for offset, key, label, color in [
        (-0.2, "metrics", "Validation", BLUE),
        (0.2, "test_micro_f1", "Test", ORANGE),
    ]:
        scores = [100 * (d[key][-1] if key == "metrics" else d[key])
                  for d in results.values()]
        bars = ax.bar(x + offset, scores, width=0.37, color=color, label=label)
        ax.bar_label(bars, labels=[f"{s:.2f}%" for s in scores], padding=5, fontsize=9)
    ax.set_xticks(x, [name.replace(" ", "\n") for name in EXPERIMENTS.values()])
    ax.set_ylim(0, 109)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Artifact-free F1 (%)")
    ax.set_title("Seven ResNet-50 experiments · validation vs. test", pad=45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)
    fig.text(0.08, 0.035,
             "Final-epoch results · threshold 0.5 · F1 measures class 1 (artifact-free), not micro-F1.\n"
             "Face cropping excludes images with no detected face; evaluation coverage may differ.",
             fontsize=10, color="#64748B")
    fig.subplots_adjust(bottom=0.22, top=0.79)
    save_figure(fig, output_dir, "model_comparison")


def plot_learning_curves(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, ("baseline", "data-augmentation", "oversampling")):
        data = results[name]
        epochs = np.arange(1, len(data["train_loss"]) + 1)
        for key, label, color in [("train_loss", "Train", BLUE),
                                  ("valid_loss", "Validation", ORANGE)]:
            ax.plot(epochs, data[key], color=color, label=label, lw=2, marker="o", ms=3)
        ax.set_title(EXPERIMENTS[name])
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.set_xticks([e for e in (1, 5, 10, 15) if e <= len(epochs)])
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Binary cross-entropy loss (log scale)")
    fig.suptitle("Learning dynamics", fontsize=18, fontweight="bold", y=1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=2, frameon=False)
    fig.text(0.08, 0.015, "Shared axes · identical loss function across these three experiments",
             fontsize=10, color="#64748B")
    fig.tight_layout(rect=(0, 0.045, 1, 0.9))
    save_figure(fig, output_dir, "learning_curves")


def run_inference(args, results):
    try:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as exc:
        raise RuntimeError(
            "Inference figures require torch and torchvision. Install them with "
            "'python -m pip install torch torchvision', or use --metrics-only."
        ) from exc

    paths = sorted(args.data_dir.glob("*.png"))
    if not paths:
        raise ValueError(f"No PNG test images found in {args.data_dir}")
    targets = np.asarray([int(p.stem.rsplit("_", 1)[-1]) for p in paths])
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Test filenames must end in _0.png or _1.png")
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    device = torch.device(args.device)
    predictions = {}
    for name in ("baseline", "oversampling"):
        checkpoint = args.runs_dir / name / "last.pt"
        # weights=None avoids downloading ImageNet weights; last.pt supplies all weights.
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device).eval()
        batches = []
        print(f"Evaluating {EXPERIMENTS[name]} on {len(paths)} test images ({device})...", flush=True)
        with torch.inference_mode():
            for start in range(0, len(paths), args.batch_size):
                images = []
                for path in paths[start:start + args.batch_size]:
                    with Image.open(path) as image:
                        images.append(transform(image.convert("RGB")))
                batch = torch.stack(images).to(device)
                batches.append(model(batch).sigmoid().flatten().cpu().numpy())
        probabilities = np.concatenate(batches)
        predicted = (probabilities >= 0.5).astype(int)
        tp = ((predicted == 1) & (targets == 1)).sum()
        denominator = (predicted == 1).sum() + (targets == 1).sum()
        actual_f1 = float(2 * tp / denominator) if denominator else 0.0
        expected_f1 = results[name]["test_micro_f1"]
        if not np.isclose(actual_f1, expected_f1, atol=1e-6, rtol=0):
            raise ValueError(
                f"{name}: reproduced test F1 {actual_f1:.8f} differs from saved "
                f"{expected_f1:.8f}. Check the test dataset and last.pt checkpoint."
            )
        print(f"Verified saved test F1: {actual_f1:.6f}")
        predictions[name] = (predicted, probabilities)
        del model
    return paths, targets, predictions


def plot_confusion_matrices(targets, predictions, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), layout="constrained")
    for ax, (name, (predicted, _)) in zip(axes, predictions.items()):
        counts = np.zeros((2, 2), dtype=int)
        np.add.at(counts, (targets, predicted), 1)
        totals = counts.sum(axis=1, keepdims=True)
        percentages = np.divide(counts * 100.0, totals, out=np.zeros((2, 2)), where=totals != 0)
        heatmap = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{counts[row, col]}\n{percentages[row, col]:.1f}%",
                        ha="center", va="center", fontsize=15,
                        color="white" if percentages[row, col] > 55 else "#172033")
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(EXPERIMENTS[name], pad=12)
    fig.colorbar(heatmap, ax=axes, shrink=0.7, label="Percentage of true class")
    fig.suptitle(f"Test errors by class · {len(targets)} images\n"
                 "Cell labels: count and row percentage · final checkpoints · threshold 0.5",
                 fontsize=13)
    save_figure(fig, output_dir, "confusion_matrices")


def plot_prediction_gallery(paths, targets, baseline_predictions, output_dir):
    predicted, probabilities = baseline_predictions
    confidence = np.where(predicted == 1, probabilities, 1 - probabilities)
    correct = predicted == targets
    selected = []
    # Reserve examples from both classes, then prioritize confident mistakes.
    for label in (0, 1):
        candidates = np.flatnonzero(correct & (targets == label))
        selected.extend(sorted(candidates, key=lambda i: (-confidence[i], i))[:2])
    errors = sorted(np.flatnonzero(~correct), key=lambda i: (-confidence[i], i))
    selected.extend(errors[:6 - len(selected)])
    remaining = sorted((i for i in range(len(paths)) if i not in selected),
                       key=lambda i: (-confidence[i], i))
    selected.extend(remaining[:6 - len(selected)])
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, selected):
        with Image.open(paths[index]) as image:
            ax.imshow(image.convert("RGB"))
        status = "Correct" if correct[index] else "Error"
        ax.set_title(f"{status} · True: {CLASS_NAMES[targets[index]]}\n"
                     f"Predicted: {CLASS_NAMES[predicted[index]]} ({confidence[index]:.1%})",
                     fontsize=11, color="#166534" if correct[index] else "#B91C1C")
        ax.text(0.5, -0.04, paths[index].name, transform=ax.transAxes,
                ha="center", fontsize=8, color="#64748B")
    fig.suptitle("Baseline predictions on test images", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.02, "Selected correct examples and highest-confidence mistakes · "
             "probabilities are uncalibrated model outputs", ha="center", fontsize=10, color="#64748B")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=3)
    save_figure(fig, output_dir, "prediction_gallery", formats=("png",))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "test",
                        help="Directory containing labeled test PNGs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets" / "results")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    configure_style()
    try:
        results = read_results(args.runs_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(results, args.output_dir)
        plot_learning_curves(results, args.output_dir)
        if not args.metrics_only:
            paths, targets, predictions = run_inference(args, results)
            plot_confusion_matrices(targets, predictions, args.output_dir)
            plot_prediction_gallery(paths, targets, predictions["baseline"], args.output_dir)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
