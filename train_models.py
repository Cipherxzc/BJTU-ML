import argparse
import subprocess

def main():
    available_models = ["Adaboost", "HierarchicalClustering", "Linear", "NaiveBayes", "SimCLR", "ResNet"]

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
        default="",
        help="Path to save the trained model."
    )
    args = parser.parse_args()

    selected_model = args.model

    model_default_save_path = {
        "Adaboost": "models/adaboost_model.pkl",
        "HierarchicalClustering": "models/hierarchical_clustering_model.safetensors",
        "Linear": "models/linear_pca_model.pkl",
        "NaiveBayes": "models/naive_bayes_model.pkl",
        "SimCLR": "models/simclr_model.safetensors",
        "ResNet": "models/resnet_model.safetensors"
    }

    if not args.model_save_path:
        args.model_save_path = model_default_save_path[selected_model]

    model_to_module = {
        "Adaboost": "src.Adaboost.train_adaboost",
        "HierarchicalClustering": "src.HierarchicalClustering.train_hier_clustering",
        "Linear": "src.Linear.train_linear",
        "NaiveBayes": "src.NaiveBayes.train_naive_bayes",
        "SimCLR": "src.SimCLR.train_simclr",
        "ResNet": "src.ResNet.train_resnet"
    }

    script_module = model_to_module[selected_model]

    subprocess.run(["python", "-m", script_module, "--train-data", args.train_data, "--test-data", args.test_data, "--model-save-path", args.model_save_path], check=True)

if __name__ == "__main__":
    main()