# MNIST with PyTorch: MLP & CNN for Beginners

一个面向深度学习初学者的 PyTorch MNIST 手写数字分类项目。

本项目分别使用：

- 多层感知机（MLP）
- 卷积神经网络（CNN）

完成从数据读取、预处理、模型定义、训练，到生成 Kaggle 提交文件的完整流程。

代码结构简洁，适合刚开始学习 PyTorch、神经网络和图像分类的同学阅读与实践。

## 模型成绩

以下为作者提供的 Kaggle 成绩：

| 模型 | Kaggle Accuracy |
|---|---:|
| MLP | 0.9825 |
| CNN | 0.9895 |

CNN 能够保留并利用图像的空间结构，因此在本项目中的表现优于 MLP。

> 上述分数是 Kaggle 提交成绩，不是代码内置验证集计算结果。当前脚本没有单独划分验证集。

## 项目结构

```text
mnist-pytorch-for-beginners/
├── Mnist_model/
│   ├── Mnist_kaggle_MLP.py
│   └── Mnist_kaggle_CNN.py
├── mnist_kaggle.npz
└── README.md
```

文件说明：

- `Mnist_kaggle_MLP.py`：使用全连接神经网络进行分类。
- `Mnist_kaggle_CNN.py`：使用卷积神经网络进行分类。
- `mnist_kaggle.npz`：经过整理的训练集与测试集。
- `submission.csv`：运行任一模型后生成的 Kaggle 预测文件。

## 数据集

数据来源：

> Eric Li. MNIST HW. Kaggle, 2021.

比赛页面：[MNIST HW – Kaggle](https://www.kaggle.com/competitions/mnist-sai)

`mnist_kaggle.npz` 包含以下数组：

| 数组 | 形状 | 内容 |
|---|---|---|
| `x_train` | `(60000, 28, 28)` | 训练图像 |
| `y_train` | `(60000,)` | 训练标签，类别为 0–9 |
| `x_test` | `(10000, 28, 28)` | 测试图像 |

图像像素值范围为 `0–255`。两个脚本都会将图像转换为浮点数，并除以 `255.0`，将像素归一化到 `0–1`。

请遵守 Kaggle 比赛页面所列的数据使用条款。

## 神经网络运行流程

两个示例都遵循相同的基本流程：

```text
读取数据
   ↓
调整输入形状并归一化
   ↓
创建 TensorDataset 和 DataLoader
   ↓
定义神经网络
   ↓
设置损失函数和优化器
   ↓
前向传播
   ↓
计算损失
   ↓
反向传播并更新参数
   ↓
在测试集上预测
   ↓
生成 submission.csv
```

这套流程也是许多 PyTorch 深度学习项目的基础结构。


## 环境要求

推荐使用 Python 3.10 或更高版本。

安装依赖：

```bash
pip install torch numpy pandas
```

主要依赖：

- PyTorch
- NumPy
- pandas

## 运行方法

请先进入仓库根目录。

运行 MLP：

```bash
python Mnist_model/Mnist_kaggle_MLP.py
```

运行 CNN：

```bash
python Mnist_model/Mnist_kaggle_CNN.py
```

训练完成后，脚本会在当前目录生成：

```text
submission.csv
```

提交文件包含两列：

```text
Id,Category
```

其中：

- `Id`：测试样本编号。
- `Category`：模型预测的数字类别。

> 两个脚本都会生成名为 `submission.csv` 的文件。运行第二个模型时，可能覆盖第一个模型生成的结果，请提前重命名需要保留的文件。

## 初学者可以学到什么？

通过阅读和运行本项目，你可以了解：

- 如何使用 NumPy 加载数据。
- 如何将 NumPy 数组转换为 PyTorch Tensor。
- 为什么需要对图像像素进行归一化。
- 如何使用 `TensorDataset` 和 `DataLoader` 分批读取数据。
- 如何继承 `torch.nn.Module` 定义模型。
- MLP 与 CNN 的输入格式和结构差异。
- 什么是前向传播、损失函数与反向传播。
- 如何使用 Adam 优化器更新模型参数。
- 如何切换到评估模式并关闭梯度计算。
- 如何将预测结果保存为 Kaggle 提交文件。

## 建议的学习顺序

如果你是第一次接触深度学习，推荐按照以下顺序学习：

1. 先运行 `Mnist_kaggle_MLP.py`。
2. 阅读 `Net` 类和 `forward()` 方法。
3. 理解训练循环中的五个核心步骤：

   ```python
   y_pred = model(x)
   loss = criterion(y_pred, y)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   ```

4. 再运行 `Mnist_kaggle_CNN.py`。
5. 对比两个模型的数据形状与网络结构。
6. 尝试修改 epoch、batch size 或 learning rate，观察结果变化。

## 可以继续尝试的改进

完成基础代码后，可以尝试：

- 自动选择 CPU、CUDA 或 Apple Silicon MPS。
- 划分训练集和验证集。
- 记录每个 epoch 的平均 loss 与 accuracy。
- 添加 Dropout 或 Batch Normalization。
- 保存和加载训练完成的模型。
- 固定随机种子，提高实验可复现性。
- 绘制训练曲线和混淆矩阵。
- 分别保存 MLP 与 CNN 的提交文件。
- 将训练、评估和预测拆分为独立函数。

## 致谢

感谢 Eric Li 在 Kaggle 发布的 [MNIST HW](https://www.kaggle.com/competitions/mnist-sai) 数据与比赛。

本项目用于深度学习与 PyTorch 入门学习。
