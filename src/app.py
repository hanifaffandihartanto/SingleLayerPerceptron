import argparse
from pathlib import Path
from prepare_data import read_dataset

def main():
    parser = argparse.ArgumentParser(
        description="SLP Sigmoid dengan file training dan validation terpisah."
    )
    parser.add_argument(
        "--train",
        default="../data/iris-training.xlsx",
        help="File Excel training (default: data/iris-training.xlsx).",
    )
    parser.add_argument(
        "--validation",
        default="../data/iris-validation.xlsx",
        help="File Excel validation (default: data/iris-validation.xlsx).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--output", default="result")
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
    
    print(training_data)
    print(validation_data)
        

if __name__ == "__main__":
    main()
