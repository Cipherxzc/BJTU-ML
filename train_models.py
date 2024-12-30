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
        default="",
        help="Path to the training data CSV file."
    )
    parser.add_argument(
        "--test-data",
        default="",
        help="Path to the testing data CSV file."
    )
    parser.add_argument(
        "--model-save-path",
        default="",
        help="Path to save the trained model."
    )
    parser.add_argument(
        "--mode",
        choices=["1", "2"],
        help="Select the training mode for non-SimCLR and non-ResNet models."
    )
    args = parser.parse_args()

    selected_model = args.model

    model_info = {
        "Adaboost": {
            "save_path": ["models/adaboost_model_1.pkl", "models/adaboost_model_2.pkl"],
            "module": "src.Adaboost.train_adaboost"
        },
        "HierarchicalClustering": {
            "save_path": ["models/hierarchical_clustering_model_1.safetensors", "models/hierarchical_clustering_model_2.safetensors"],
            "module": "src.HierarchicalClustering.train_hier_classifier"
        },
        "Linear": {
            "save_path": ["models/linear_1", "models/linear_2"],
            "module": "src.Linear.train_linear"
        },
        "NaiveBayes": {
            "save_path": ["models/naive_bayes_model_1.safetensors", "models/naive_bayes_model_2.safetensors"],
            "module": "src.NaiveBayes.train_naive_bayes"
        },
        "SimCLR": {
            "save_path": ["models/simclr_model.safetensors"],
            "module": "src.SimCLR.train_simclr"
        },
        "ResNet": {
            "save_path": ["models/resnet_model.safetensors"],
            "module": "src.ResNet.train_resnet"
        }
    }

    if selected_model in ["SimClr", "ResNet"]:
        train_data = args.train_data if args.train_data else "data/fashion-mnist_train.csv"
        test_data = args.test_data if args.test_data else "data/fashion-mnist_test.csv"
        save_path = args.model_save_path if args.model_save_path else model_info[selected_model]["save_path"][0]
    else:
        if not args.mode:
            parser.error("Training mode must be specified for non-SimCLR and non-ResNet models.")
        
        train_data = args.train_data if args.train_data else f"data/features{args.mode}_train.csv"
        test_data = args.test_data if args.test_data else f"data/features{args.mode}_test.csv"
        save_path = args.model_save_path if args.model_save_path else model_info[selected_model]["save_path"][int(args.mode)-1]

    script_module = model_info[selected_model]["module"]

    subprocess.run(["python", "-m", script_module, "--train-data", train_data, "--test-data", test_data, "--model-save-path", save_path], check=True)

if __name__ == "__main__":
    main()