# 项目代码运行流程

## 流程图示

\[
\begin{array}{c}
\text{FashionMNIST} \\
\downarrow \\
\begin{array}{cc}
\text{SimCLR(自监督) 处理} & \text{ResNet(有监督) 处理} \\
\downarrow & \downarrow \\
\text{生成特征集1 (features1)} & \text{生成特征集2 (features2)} \\
\end{array} \\
\downarrow \\
\begin{array}{c}
\text{使用 Feature Sets 训练模型} \\
\begin{array}{cc}
\text{NaiveBayes (features1)} & \text{NaiveBayes (features2)} \\
\text{Adaboost (features1)} & \text{Adaboost (features2)} \\
\vdots & \vdots \\
\end{array} \\
\end{array} \\
\downarrow \\
\text{使用 Test Set 评估模型}
\end{array}
\]

## 详细步骤

1. **数据预处理**
   - **原始数据集**: 使用 FashionMNIST 数据集。
   - **SimCLR 处理**: 将原始图像转换为特征格式，生成 `features1` 数据集。
   - **ResNet 处理**: 将原始图像转换为特征格式，生成 `features2` 数据集。

2. **模型训练**
   - 使用 `features1` 数据集训练若干模型（如NaiveBayes, Adaboost, ...）。
   - 使用 `features2` 数据集训练若干模型（如NaiveBayes, Adaboost, ...）。

3. **模型评估**
   - 使用测试集对所有训练好的模型进行统一评估。
