"""Summarise a training run: best metrics, epoch progression and loss/mAP/F1 plots."""

from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from helmet_detection.config import base_parser, load_config, resolve


def main() -> None:
    parser = base_parser("Summarise a YOLOv8 training run.")
    parser.add_argument("--run-name", help="override train.run_name")
    args = parser.parse_args()
    cfg = load_config(args.config)

    run_name = args.run_name or cfg["train"]["run_name"]
    run_dir = str(resolve(cfg["paths"]["runs_dir"]) / "helmet" / run_name)
    report_dir = str(resolve(cfg["paths"]["reports_dir"]) / f"{run_name}_training")
    csv_path = os.path.join(run_dir, "results.csv")

    os.makedirs(report_dir, exist_ok=True)

    # Back up the raw CSV before doing anything else
    shutil.copy(csv_path, os.path.join(report_dir, f"{run_name}_train_results.csv"))

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # YOLO sometimes adds spaces to column names

    print("=" * 60)
    print(f"TRAINING SUMMARY — {run_name}")
    print("=" * 60)
    print(f"Total epochs:        {len(df)}")
    print(f"Best mAP50:          {df['metrics/mAP50(B)'].max():.4f} (epoch {df['metrics/mAP50(B)'].idxmax() + 1})")
    print(f"Best mAP50-95:       {df['metrics/mAP50-95(B)'].max():.4f} (epoch {df['metrics/mAP50-95(B)'].idxmax() + 1})")
    print(f"Best Precision:      {df['metrics/precision(B)'].max():.4f} (epoch {df['metrics/precision(B)'].idxmax() + 1})")
    print(f"Best Recall:         {df['metrics/recall(B)'].max():.4f} (epoch {df['metrics/recall(B)'].idxmax() + 1})")
    print(f"Final train box_loss:{df['train/box_loss'].iloc[-1]:.4f}")
    print(f"Final train cls_loss:{df['train/cls_loss'].iloc[-1]:.4f}")
    print(f"Final val box_loss:  {df['val/box_loss'].iloc[-1]:.4f}")
    print(f"Final val cls_loss:  {df['val/cls_loss'].iloc[-1]:.4f}")

    print("\n" + "=" * 60)
    print("EPOCH PROGRESSION (every 10 epochs)")
    print("=" * 60)
    cols = ['epoch', 'metrics/precision(B)', 'metrics/recall(B)',
            'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    milestone_df = df[df['epoch'] % 10 == 0][cols].copy()
    milestone_df.columns = ['Epoch', 'Precision', 'Recall', 'mAP50', 'mAP50-95']
    print(milestone_df.to_string(index=False))

    # F1 isn't logged by YOLO directly, so we compute it from precision and recall
    df['F1'] = 2 * (df['metrics/precision(B)'] * df['metrics/recall(B)']) / \
               (df['metrics/precision(B)'] + df['metrics/recall(B)'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Helmet Detection — {run_name} Training Results", fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train/box_loss'], label='Train box loss', color='blue')
    ax1.plot(df['epoch'], df['val/box_loss'],   label='Val box loss',   color='blue', linestyle='--')
    ax1.plot(df['epoch'], df['train/cls_loss'], label='Train cls loss', color='red')
    ax1.plot(df['epoch'], df['val/cls_loss'],   label='Val cls loss',   color='red',  linestyle='--')
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['metrics/mAP50(B)'],    label='mAP50',    color='green',  linewidth=2)
    ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='orange', linewidth=2)
    ax2.axhline(y=df['metrics/mAP50(B)'].max(), color='green', linestyle=':', alpha=0.5)
    ax2.set_title("mAP Scores")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='purple', linewidth=2)
    ax3.plot(df['epoch'], df['metrics/recall(B)'],    label='Recall',    color='brown',  linewidth=2)
    ax3.set_title("Precision & Recall")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Score")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['F1'], label='F1 Score', color='teal', linewidth=2)
    ax4.axhline(y=df['F1'].max(), color='teal', linestyle=':', alpha=0.5)
    ax4.set_title(f"F1 Score (Best: {df['F1'].max():.4f} at epoch {df['F1'].idxmax() + 1})")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("F1")
    ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_out = os.path.join(report_dir, f"{run_name}_train_analysis_plots.png")
    plt.savefig(plot_out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_out}")

    # Copy the YOLO-generated plots into the report folder with consistent names
    for src_name, dst_name in [
        ("confusion_matrix_normalized.png", f"{run_name}_train_confusion_matrix.png"),
        ("results.png",                     f"{run_name}_train_metrics.png"),
        ("BoxF1_curve.png",                 f"{run_name}_train_F1_curve.png"),
        ("BoxPR_curve.png",                 f"{run_name}_train_PR_curve.png"),
        ("BoxP_curve.png",                  f"{run_name}_train_P_curve.png"),
        ("BoxR_curve.png",                  f"{run_name}_train_R_curve.png"),
    ]:
        src = os.path.join(run_dir, src_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(report_dir, dst_name))

    print(f"All files saved to {report_dir}")


if __name__ == "__main__":
    main()
