import os
import json
import math
import logging
from typing import Dict, Any, List, Optional, Union

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    Reusable utility for parsing, plotting, and saving training metrics
    across any PEFT fine-tuning recipe (LoRA, QLoRA, Prefix Tuning, etc.).
    """

    @staticmethod
    def extract_metrics(log_history: List[Dict[str, Any]]) -> Dict[str, List[Union[int, float]]]:
        """
        Extracts step-by-step metrics from Hugging Face trainer.state.log_history.

        Args:
            log_history (List[Dict[str, Any]]): Raw log history list from Trainer.

        Returns:
            Dict[str, List]: Structured dictionary with steps, losses, perplexities, grad_norms, and learning_rates.
        """
        loss_entries = [entry for entry in log_history if "loss" in entry]
        
        steps = []
        losses = []
        perplexities = []
        grad_norms = []
        learning_rates = []
        epochs = []

        for entry in loss_entries:
            step = entry.get("step")
            loss = entry.get("loss")
            
            if step is not None and loss is not None:
                steps.append(step)
                losses.append(loss)
                # Compute step-level perplexity
                try:
                    perplexities.append(round(math.exp(loss), 3))
                except OverflowError:
                    perplexities.append(float("inf"))
                
                grad_norms.append(entry.get("grad_norm", None))
                learning_rates.append(entry.get("learning_rate", None))
                epochs.append(entry.get("epoch", None))

        return {
            "steps": steps,
            "losses": losses,
            "perplexities": perplexities,
            "grad_norms": grad_norms,
            "learning_rates": learning_rates,
            "epochs": epochs
        }

    @classmethod
    def plot_and_save_curves(
        cls,
        log_history: Union[List[Dict[str, Any]], Any],
        output_dir: str,
        filename: str = "training_curves.png",
        show_plot: bool = True
    ) -> Optional[str]:
        """
        Generates a 4-panel training dashboard (Loss, Perplexity, Grad Norm, Learning Rate)
        and saves the artifact to the model's output directory.

        Args:
            log_history: Trainer's state.log_history list or the Trainer object itself.
            output_dir (str): Directory where model adapters and artifacts are saved.
            filename (str): Name of the generated plot image.
            show_plot (bool): Whether to display the plot inline (e.g. in Colab / Jupyter).

        Returns:
            str: Path to the saved plot image.
        """
        # If a Trainer object was passed directly, extract state.log_history
        if hasattr(log_history, "state") and hasattr(log_history.state, "log_history"):
            log_history = log_history.state.log_history

        metrics = cls.extract_metrics(log_history)
        steps = metrics["steps"]

        if not steps:
            logger.warning("No training step logs found to plot.")
            return None

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)

        if HAS_MATPLOTLIB:
            # Create a 4-panel figure
            fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
            fig.suptitle("PEFT Fine-Tuning Progression Dashboard", fontsize=14, fontweight="bold", y=1.03)

            # 1. Training Loss
            axes[0].plot(steps, metrics["losses"], color="#1d4ed8", lw=2, marker="o", markersize=3, label="Training Loss")
            axes[0].set_title("Training Loss", fontweight="bold", fontsize=11)
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Cross-Entropy Loss")
            axes[0].grid(True, linestyle="--", alpha=0.5)
            axes[0].legend(loc="upper right")

            # 2. Step-Level Perplexity
            valid_ppl = [p for p in metrics["perplexities"] if p != float("inf")]
            axes[1].plot(steps, metrics["perplexities"], color="#15803d", lw=2, marker="s", markersize=3, label="Perplexity (PPL)")
            axes[1].set_title("Step-Level Perplexity (PPL)", fontweight="bold", fontsize=11)
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("PPL (exp(loss))")
            if valid_ppl:
                axes[1].set_ylim(bottom=0, top=min(max(valid_ppl) * 1.1, 100))
            axes[1].grid(True, linestyle="--", alpha=0.5)
            axes[1].legend(loc="upper right")

            # 3. Gradient Norm Stability
            has_grad_norm = any(g is not None for g in metrics["grad_norms"])
            if has_grad_norm:
                clean_gn = [g if g is not None else 0 for g in metrics["grad_norms"]]
                axes[2].plot(steps, clean_gn, color="#b45309", lw=2, marker="^", markersize=3, label="Grad Norm")
                axes[2].set_title("Gradient Norm Stability", fontweight="bold", fontsize=11)
                axes[2].set_xlabel("Step")
                axes[2].set_ylabel("Gradient Norm")
                axes[2].grid(True, linestyle="--", alpha=0.5)
                axes[2].legend(loc="upper right")
            else:
                axes[2].text(0.5, 0.5, "Grad Norm Not Logged", horizontalalignment="center", verticalalignment="center")

            # 4. Learning Rate Schedule
            has_lr = any(lr is not None for lr in metrics["learning_rates"])
            if has_lr:
                clean_lr = [lr if lr is not None else 0 for lr in metrics["learning_rates"]]
                axes[3].plot(steps, clean_lr, color="#7c3aed", lw=2, marker="x", markersize=3, label="Learning Rate")
                axes[3].set_title("Learning Rate Schedule", fontweight="bold", fontsize=11)
                axes[3].set_xlabel("Step")
                axes[3].set_ylabel("LR")
                axes[3].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
                axes[3].grid(True, linestyle="--", alpha=0.5)
                axes[3].legend(loc="upper right")
            else:
                axes[3].text(0.5, 0.5, "LR Not Logged", horizontalalignment="center", verticalalignment="center")

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"✅ Training curves successfully saved to {save_path}")

            if show_plot:
                plt.show()

            plt.close(fig)
        else:
            logger.warning("matplotlib not installed. Skipping image plot generation.")
            save_path = None

        # Always save structured JSON summary
        json_path = os.path.join(output_dir, "training_metrics.json")
        summary_data = {
            "initial_loss": metrics["losses"][0] if metrics["losses"] else None,
            "final_loss": metrics["losses"][-1] if metrics["losses"] else None,
            "initial_perplexity": metrics["perplexities"][0] if metrics["perplexities"] else None,
            "final_perplexity": metrics["perplexities"][-1] if metrics["perplexities"] else None,
            "total_logged_steps": len(steps),
            "step_history": metrics
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"✅ Training metrics JSON summary saved to {json_path}")

        return save_path


# Convenience functional alias
def plot_training_curves(
    log_history: Union[List[Dict[str, Any]], Any],
    output_dir: str,
    filename: str = "training_curves.png",
    show_plot: bool = True
) -> Optional[str]:
    """Convenience function for plotting and saving training curves."""
    return TrainingVisualizer.plot_and_save_curves(log_history, output_dir, filename, show_plot)
