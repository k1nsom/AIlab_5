import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Define paths
data_path = './'
train_file = os.path.join(data_path, 'train.txt')
test_file = os.path.join(data_path, 'test_without_label.txt')

# Load and preprocess data
def load_data(file_path, is_test=False):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            guid, tag = line.strip().split(',')
            data.append((guid, None if is_test else tag))
    return data

train_data = load_data(train_file)
test_data = load_data(test_file, is_test=True)

# Define Dataset class
class MultimodalDataset(Dataset):
    def __init__(self, data, tokenizer, transform, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        guid, label = self.data[idx]
        text = f"Sample text for {guid}"  # Placeholder, replace with actual text loading logic
        image = torch.rand(3, 224, 224)  # Placeholder, replace with actual image loading logic

        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        if self.is_test:
            return input_ids, attention_mask, image, guid

        label = {'positive': 0, 'neutral': 1, 'negative': 2}[label]
        return input_ids, attention_mask, image, label

# Initialize tokenizer and transforms
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Split data into training and validation
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

train_dataset = MultimodalDataset(train_data, tokenizer, transform)
val_dataset = MultimodalDataset(val_data, tokenizer, transform)
test_dataset = MultimodalDataset(test_data, tokenizer, transform, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the model
class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        from torchvision.models import ResNet18_Weights
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.image_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_model.fc = nn.Identity()  # Remove classification head

        self.fc = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size + 512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_model(images)
        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.fc(combined_features)
        return output

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch_idx, (input_ids, attention_mask, images, labels) in enumerate(loader):
        input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy

# Validation loop
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, images, labels in loader:
            input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['positive', 'neutral', 'negative'])
    return total_loss / len(loader), accuracy, report

# Training process
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy, val_report = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(val_report)

# Prediction on test data
model.eval()
results = []
with torch.no_grad():
    for input_ids, attention_mask, images, guids in test_loader:
        input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)

        outputs = model(input_ids, attention_mask, images)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        results.extend(zip(guids, preds))

# Save predictions
with open('test_predictions.txt', 'w') as f:
    f.write('guid,tag\n')
    for guid, pred in results:
        tag = ['positive', 'neutral', 'negative'][pred]
        f.write(f"{guid},{tag}\n")
