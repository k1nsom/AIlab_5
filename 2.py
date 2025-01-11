import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Define dataset
class TextImageDataset(Dataset):
    def __init__(self, csv_file, data_dir, tokenizer, max_length, transform=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        guid = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1] if not self.is_test else "null"

        # Load text
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        with open(text_path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()

        # Tokenize text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Load image
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "image": image,
            "label": label if label == "null" else int(label == "positive") * 2 + int(label == "neutral")
        }

# Define model
class TextImageClassifier(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", num_classes=3):
        super(TextImageClassifier, self).__init__()
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_model.fc = nn.Identity()  # Remove final layer
        self.text_fc = nn.Linear(768, 256)
        self.image_fc = nn.Linear(2048, 256)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_features = self.text_fc(text_features)

        image_features = self.image_model(images)
        image_features = self.image_fc(image_features)

        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.classifier(combined_features)
        return output

# Training and evaluation functions
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)

            outputs = model(input_ids, attention_mask, images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            predictions.extend(preds)

    return predictions

# Main script
def main():
    data_dir = "./data"
    train_csv = "train.txt"
    test_csv = "test_without_label.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare datasets and dataloaders
    train_dataset = TextImageDataset(train_csv, data_dir, tokenizer, max_length=128, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = TextImageDataset(test_csv, data_dir, tokenizer, max_length=128, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = TextImageClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Train model
    for epoch in range(5):  # Adjust epochs as needed
        print(f"Epoch {epoch + 1}")
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Loss: {train_loss:.4f}")

    # Evaluate on test set
    predictions = evaluate_model(model, test_loader, device)

    # Save predictions
    test_data = pd.read_csv(test_csv)
    test_data["tag"] = ["positive" if p == 2 else "neutral" if p == 1 else "negative" for p in predictions]
    test_data.to_csv("test_predictions.csv", index=False)

if __name__ == "__main__":
    main()
