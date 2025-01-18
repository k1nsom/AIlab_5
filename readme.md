# 文本-图像分类项目

本项目实现了一个多模态文本-图像分类任务，使用BERT进行文本处理，使用ResNet-50进行图像处理。该模型将文本和图像对进行分类，分为三类：正面、负面、中性。

## 需求

运行本项目需要以下Python库：

```bash
pip install -r requirements.txt
```

## 代码文件结构

```
项目目录/
│
├── main.py  # 训练和评估模型的主代码文件
├── test_prediction.csv  # 测试集预测结果输出文件
├── test_without_label.txt  # 无标签的测试集文件
├── train.txt  # 训练集文件
└── data/  # 存放图片和文本数据的文件夹
    ├── 1.jpg  # 示例图片文件
    └── 1.txt  # 示例文本文件
```

### 完整的执行流程

1. 克隆仓库：

   ```bash
   git clone https://github.com/k1nsom/AIlab_5.git
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 运行主脚本进行模型训练和评估：

   ```bash
   python main.py
   ```

4. 训练完成后，预测结果将保存在 `test_predictions.csv` 文件中。

## 参考的代码库

本项目使用了以下库：

- `torch`：用于构建和训练深度学习模型。
- `pandas`：用于处理数据集（例如读取和写入CSV文件）。
- `transformers`：用于加载和使用预训练的BERT模型进行文本处理。
- `torchvision`：用于图像数据处理和加载预训练的ResNet-50模型。
- `Pillow`：用于图像处理任务。
- `tqdm`：用于显示训练和评估过程中的进度条。

## 参考资料

[博客](https://github.com/YeexiaoZheng/Multimodal-Sentiment-Analysis),[仓库](https://github.com/YeexiaoZheng/Multimodal-Sentiment-Analysis)
