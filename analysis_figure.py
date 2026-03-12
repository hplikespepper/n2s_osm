import argparse
import json
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


def load_metrics(jsonl_path: Path):
    epochs = []
    avg_init_distance = []
    avg_init_makespan = []
    avg_final_distance = []
    avg_final_makespan = []
    avg_final_total_cost = []
    avg_total_reward = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            epochs.append(record["epoch"])
            avg_init_distance.append(record["avg_init_distance"])
            avg_init_makespan.append(record["avg_init_makespan"])
            avg_final_distance.append(record["avg_final_distance"])
            avg_final_makespan.append(record["avg_final_makespan"])
            avg_final_total_cost.append(record["avg_final_total_cost"])
            avg_total_reward.append(record["avg_total_reward"])

    return {
        "epochs": epochs,
        "avg_init_distance": avg_init_distance,
        "avg_init_makespan": avg_init_makespan,
        "avg_final_distance": avg_final_distance,
        "avg_final_makespan": avg_final_makespan,
        "avg_final_total_cost": avg_final_total_cost,
        "avg_total_reward": avg_total_reward,
    }


def plot_figures(metrics, save_dir: Optional[Path]):
    epochs = metrics["epochs"]
    avg_init_total_cost = [
        init_dist + init_makespan
        for init_dist, init_makespan in zip(
            metrics["avg_init_distance"], metrics["avg_init_makespan"]
        )
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["avg_final_distance"], label="Avg distance cost")
    plt.plot(epochs, metrics["avg_final_makespan"], label="Avg makespan cost")
    plt.plot(epochs, metrics["avg_final_total_cost"], label="Avg total cost")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Avg cost over epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir / "figure1_distance_makespan_total_cost.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["avg_init_distance"], label="Avg init distance cost")
    plt.plot(epochs, metrics["avg_final_distance"], label="Avg last distance cost")
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.title("Init vs last distance cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir / "figure2_init_vs_last_distance.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["avg_init_makespan"], label="Avg init makespan cost")
    plt.plot(epochs, metrics["avg_final_makespan"], label="Avg last makespan cost")
    plt.xlabel("Epoch")
    plt.ylabel("Makespan")
    plt.title("Init vs last makespan cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir / "figure3_init_vs_last_makespan.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["avg_total_reward"], label="Avg total reward")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Average reward over epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir / "figure4_avg_total_reward.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_init_total_cost, label="Avg init total cost")
    plt.plot(epochs, metrics["avg_final_total_cost"], label="Avg last total cost")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Init total cost vs last total cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir / "figure5_init_total_cost_last_cost.png", dpi=300, bbox_inches="tight")

    plt.tight_layout()
    if save_dir is None:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MVPDTSP training metrics.")
    parser.add_argument(
        "--result_file",
        type=Path,
        default=Path(
            "n2s_osm_marl/outputs/mvpdtsp_20/"
            "mvpdtsp20_makespan_log_20260130T200924/mvpdtsp_epoch_metrics.jsonl"
        ),
        help="Path to metrics jsonl file.",
    )
    parser.add_argument(
        "--save_figure",
        type=Path,
        default=None,
        help="Directory to save figures. If not set, figures are shown only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = load_metrics(args.result_file)
    save_dir = None
    if args.save_figure is not None:
        save_dir = args.save_figure
        save_dir.mkdir(parents=True, exist_ok=True)
    plot_figures(metrics, save_dir)


if __name__ == "__main__":
    main()
