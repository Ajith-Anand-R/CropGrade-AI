import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Dataset" / "fgrade" / "data"
MODEL_SAVE_PATH = BASE_DIR / "Models" / "classifier.pth"

def train_classifier():
    train_path = DATA_DIR / "Training_set"
    test_path = DATA_DIR / "Testing_set"

    if not train_path.exists() or not test_path.exists():
        print(f"Error: FGrade dataset folders not found at {DATA_DIR}. Please run download_datasets.py first.")
        return

    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    print("Loading FGrade datasets...")
    train_dataset = datasets.ImageFolder(root=str(train_path), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=str(test_path), transform=test_transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)
    print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} testing images.")
    print(f"Number of classes (freshness levels 1-10): {num_classes}")

    # Load ResNet18 model (Pretrained on ImageNet)
    print("Initializing ResNet18 model...")
    # ResNet18 is faster and lighter than VGG16, making it excellent for CPU/MVP deployment
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / len(test_loader.dataset)
        val_epoch_acc = 100. * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")

    # Save model weights
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Classifier model weights saved successfully at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_classifier()
