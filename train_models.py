import argparse
import os
import subprocess

def main():
    available_models = ["Adaboost", "HierarchicalClustering", "Linear", "NaiveBayes", "SimCLR"]

    parser = argparse.ArgumentParser(description="Select models for training.")
    parser.add_argument(
        "-m", "--model",
        choices=available_models,
        required=True,
        help=f"Select one model to train. Available models: {', '.join(available_models)}"
    )
    parser.add_argument(
        "--train-data",
        default="data/fashion-mnist_train.csv",
        help="Path to the training data CSV file."
    )
    parser.add_argument(
        "--test-data",
        default="data/fashion-mnist_test.csv",
        help="Path to the testing data CSV file."
    )
    parser.add_argument(
        "--model-save-path",
        default="models/naive_bayes_model.pkl",
        help="Path to save the trained model."
    )
    args = parser.parse_args()

    selected_model = args.model

    model_to_script = {
        "Adaboost": "train_adaboost.py",
        "HierarchicalClustering": "train_hier_clustering.py",
        "Linear": "train_linear.py",
        "NaiveBayes": "train_naive_bayes.py",
        "SimCLR": "train_simclr.py"
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, 'src', selected_model, model_to_script[selected_model])

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Training script for {selected_model} not found at {script_path}")

    subprocess.run(["python", script_path, "--train-data", args.train_data, "--test-data", args.test_data, "--model-save-path", args.model_save_path], check=True)

if __name__ == "__main__":
    main()