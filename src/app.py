import argparse
from pathlib import Path
from prepare_data import read_dataset
from single_layer_perceptron import training_validation
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="SLP Sigmoid dengan file training dan validation terpisah."
    )
    parser.add_argument(
        "--train",
        default="../data/iris-training.xlsx",
        help="File Excel training (default: ../data/iris-training.xlsx).",
    )
    parser.add_argument(
        "--validation",
        default="../data/iris-validation.xlsx",
        help="File Excel validation (default: ../data/iris-validation.xlsx).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--output", default="../result")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("Epoch minimal 1.")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate harus lebih besar dari 0.")

    base_dir = Path(__file__).resolve().parent

    def resolve_file(argument):
        p = Path(argument).expanduser()
        if not p.is_absolute():
            p = base_dir / p
        return p.resolve()

    train_path = resolve_file(args.train)
    validation_path = resolve_file(args.validation)

    training_data = read_dataset(train_path)
    validation_data = read_dataset(validation_path)
    
    # parameter awal untuk bias, theta1, theta2, theta3 dan theta4
    weights = [0.5, 0.5, 0.5, 0.5, 0.5]
    summary = training_validation(training_data, validation_data, args.epochs, weights, args.learning_rate)

    print("\n========== SINGLE LAYER PERCEPTRON (SIGMOID) ==========")
    print(f"File training    : {train_path}")
    print(f"File validation  : {validation_path}")
    print(f"Jumlah training  : {len(training_data)}")
    print(f"Jumlah validation: {len(validation_data)}")
    print(f"Learning rate    : {args.learning_rate}")
    print(f"Bobot awal       : {weights}")
    print(f"Epoch            : {args.epochs}\n")
    print("Epoch | Train Accuracy | Train Loss | Validation Accuracy | Validation Loss")
    print("-" * 78)
    
    for x in summary:
        print(
            f"{x['epoch']:>5} | "
            f"{x['training_accuracy']*100:>13.2f}% | "
            f"{x['training_mse']:>9.6f} | "
            f"{x['validation_accuracy']*100:>18.2f}% | "
            f"{x['validation_mse']:>14.6f}"
        )

    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # CHART Training dan Validation Accuracy
    epoch_numbers = [item["epoch"] for item in summary]
    train_accuracy_values = [item["training_accuracy"] * 100 for item in summary]
    val_accuracy_values = [item["validation_accuracy"] * 100 for item in summary]

    plt.figure(figsize=(9, 5))
    plt.plot(
        epoch_numbers, train_accuracy_values,
        marker="o", linewidth=2, label="Training Accuracy"
    )
    plt.plot(
        epoch_numbers, val_accuracy_values,
        marker="s", linewidth=2, label="Validation Accuracy"
    )
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(epoch_numbers)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_training_validation.png", dpi=300)
    plt.close()

    # CHART Training dan Validation Loss
    train_loss_values = [item["training_mse"] for item in summary]
    val_loss_values = [item["validation_mse"] for item in summary]

    plt.figure(figsize=(9, 5))
    plt.plot(
        epoch_numbers, train_loss_values,
        marker="o", linewidth=2, label="Training Loss"
    )
    plt.plot(
        epoch_numbers, val_loss_values,
        marker="s", linewidth=2, label="Validation Loss"
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epoch_numbers)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_training_validation.png", dpi=300)
    plt.close()

    print(f"\nFile hasil tersimpan di: {output_dir.resolve()}")
    print("- accuracy_training_validation.png")
    print("- loss_training_validation.png")


if __name__ == "__main__":
    main()
