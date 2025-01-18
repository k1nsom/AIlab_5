import os  # 导入os库，用于与操作系统进行交互（例如文件路径操作）
import pandas as pd  # 导入pandas库，用于数据处理（如读取CSV文件）
import torch  # 导入PyTorch库，提供深度学习的核心功能
from torch.utils.data import Dataset, DataLoader  # 从PyTorch中导入Dataset和DataLoader，用于自定义数据集和批量加载
from transformers import BertTokenizer, BertModel  # 导入HuggingFace的transformers库，用于加载BERT模型和Tokenizer
from torchvision import transforms, models  # 导入torchvision库，用于图像处理和使用预训练的图像模型
import torch.nn as nn  # 导入PyTorch的神经网络模块，用于构建深度学习模型
from PIL import Image  # 导入PIL库，用于处理图像数据
from tqdm import tqdm  # 导入tqdm库，用于显示训练和评估进度条

# 定义自定义数据集类
class TextImageDataset(Dataset):
    def __init__(self, csv_file, data_dir, tokenizer, max_length, transform=None, is_test=False):
        # 初始化时加载数据集的CSV文件，并保存相关参数
        self.data = pd.read_csv(csv_file)  # 读取CSV文件
        self.data_dir = data_dir  # 数据目录，用于存储图像和文本文件
        self.tokenizer = tokenizer  # BERT的tokenizer，用于处理文本
        self.max_length = max_length  # 最大文本长度
        self.transform = transform  # 图像处理操作
        self.is_test = is_test  # 是否为测试集，测试集不需要标签

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引的样本
        guid = self.data.iloc[idx, 0]  # 获取文本和图像的唯一标识符
        label = self.data.iloc[idx, 1] if not self.is_test else "null"  # 如果是测试集，没有标签

        # 加载文本数据
        text_path = os.path.join(self.data_dir, f"{guid}.txt")  # 构建文本文件路径
        with open(text_path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()  # 读取文本内容

        # 使用BERT的tokenizer将文本转换为模型可以接受的格式
        inputs = self.tokenizer(
            text,  # 文本
            max_length=self.max_length,  # 最大长度
            padding="max_length",  # 填充到最大长度
            truncation=True,  # 超过最大长度时截断
            return_tensors="pt"  # 返回PyTorch的张量
        )

        # 加载图像数据
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")  # 构建图像文件路径
        image = Image.open(image_path).convert("RGB")  # 打开图像并转换为RGB格式
        if self.transform:
            image = self.transform(image)  # 如果有图像处理操作，则进行处理

        # 返回处理后的数据，包含文本和图像的输入，以及标签
        return {
            "input_ids": inputs["input_ids"].squeeze(0),  # 文本的输入ID
            "attention_mask": inputs["attention_mask"].squeeze(0),  # 文本的attention mask
            "image": image,  # 图像数据
            "label": label if label == "null" else int(label == "positive") * 2 + int(label == "neutral")  # 标签转换为数字，positive=2, neutral=1, negative=0
        }

# 定义模型类
class TextImageClassifier(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", num_classes=3):
        super(TextImageClassifier, self).__init__()
        # 初始化时加载文本模型（BERT）和图像模型（ResNet-50）
        self.text_model = BertModel.from_pretrained(text_model_name)  # 加载预训练BERT模型
        self.image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 加载预训练ResNet-50模型
        self.image_model.fc = nn.Identity()  # 移除ResNet的最后一层全连接层
        self.text_fc = nn.Linear(768, 256)  # 文本部分的全连接层
        self.image_fc = nn.Linear(2048, 256)  # 图像部分的全连接层
        self.classifier = nn.Linear(512, num_classes)  # 最后的分类层，输入维度为512（256 + 256）

    def forward(self, input_ids, attention_mask, images):
        # 前向传播函数，接收文本和图像输入并进行处理
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # 获取文本特征
        text_features = self.text_fc(text_features)  # 通过全连接层进行处理

        image_features = self.image_model(images)  # 获取图像特征
        image_features = self.image_fc(image_features)  # 通过图像部分的全连接层处理

        # 将文本和图像特征进行拼接
        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.classifier(combined_features)  # 通过分类层得到最终输出
        return output

# 训练和评估函数
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()  # 将模型设为训练模式
    total_loss = 0  # 初始化总损失
    for batch in tqdm(dataloader, desc="Training", leave=False):  # 遍历数据加载器中的每个批次
        # 将数据加载到设备（GPU/CPU）
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()  # 清空之前的梯度
        outputs = model(input_ids, attention_mask, images)  # 获取模型输出
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()  # 累加损失

    return total_loss / len(dataloader)  # 返回平均损失

def evaluate_model(model, dataloader, device):
    model.eval()  # 将模型设为评估模式
    predictions = []  # 用于保存预测结果
    with torch.no_grad():  # 关闭梯度计算（评估时不需要计算梯度）
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):  # 遍历评估集
            # 将数据加载到设备（GPU/CPU）
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)

            outputs = model(input_ids, attention_mask, images)  # 获取模型输出
            preds = torch.argmax(outputs, dim=1).cpu().tolist()  # 获取预测类别
            predictions.extend(preds)  # 将预测结果添加到列表中

    return predictions  # 返回所有预测结果

# 主脚本
def main():
    data_dir = "./data"  # 数据目录
    train_csv = "train.txt"  # 训练集CSV文件
    test_csv = "test_without_label.txt"  # 测试集CSV文件

    # 检查是否有GPU，若没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # 加载BERT的tokenizer
    transform = transforms.Compose([  # 图像预处理操作
        transforms.Resize((224, 224)),  # 将图像大小调整为224x224
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图像进行归一化
    ])

    # 准备数据集和数据加载器
    train_dataset = TextImageDataset(train_csv, data_dir, tokenizer, max_length=128, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 训练集数据加载器

    test_dataset = TextImageDataset(test_csv, data_dir, tokenizer, max_length=128, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # 测试集数据加载器

    # 初始化模型、损失函数和优化器
    model = TextImageClassifier().to(device)
    criterion = nn.CrossEntropyLoss()  # 分类问题常用的损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # 使用Adam优化器

    # 训练模型
    for epoch in range(5):  # 设置训练轮数
        print(f"Epoch {epoch + 1}")
        train_loss = train_model(model, train_loader, criterion, optimizer, device)  # 训练一个epoch
        print(f"Loss: {train_loss:.4f}")

    # 在测试集上进行评估
    predictions = evaluate_model(model, test_loader, device)

    # 保存预测结果
    test_data = pd.read_csv(test_csv)  # 读取测试集数据
    test_data["tag"] = ["positive" if p == 2 else "neutral" if p == 1 else "negative" for p in predictions]
    test_data.to_csv("test_predictions.csv", index=False)  # 保存预测标签

# 运行主函数
if __name__ == "__main__":
    main()
