import argparse
import os
import subprocess

def main():
    # 定义支持的模型列表
    available_models = ["Adaboost", "HierarchicalClustering", "Linear", "NaiveBayes", "SimCLR"]

    # 使用 argparse 接收命令行参数
    parser = argparse.ArgumentParser(description="Select models for evaluation.")
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        choices=available_models,
        required=True,
        help=f"Select one or more models to evaluate. Available models: {', '.join(available_models)}"
    )
    args = parser.parse_args()

    # 获取用户选择的模型列表
    selected_models = args.models

    # 提示用户选择的模型
    print("You have selected the following models for evaluation:")
    for model in selected_models:
        print(f" - {model}")

    # 遍历用户选择的模型，调用对应文件夹中的 model_evaluation.py
    for model in selected_models:
        print(f"\nStarting evaluation for {model} model...")
        model_script_path = os.path.join(model, "model_evaluation.py")

        # 检查对应文件夹中的 model_evaluation.py 是否存在
        if not os.path.exists(model_script_path):
            print(f"Error: The evaluation script for {model} does not exist at {model_script_path}. Skipping...")
            continue

        # 调用子进程运行对应的 model_evaluation.py，并设置 cwd 为模型文件夹
        try:
            subprocess.run(["python", "model_evaluation.py"], cwd=model, check=True)
            print(f"Evaluation for {model} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: An issue occurred while evaluating {model}.")
            print(f"Details: {e}")
        except Exception as e:
            print(f"Unexpected error while evaluating {model}: {e}")

    print("\nAll selected evaluations are complete.")

if __name__ == "__main__":
    main()
