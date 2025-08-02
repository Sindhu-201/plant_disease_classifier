import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pickle

# Dataset Class


class PlantDiseaseDataset(Dataset):
    """Custom Dataset for loading plant disease images"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image and apply transformations
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

# Model Architecture


class PlantDiseaseModel(nn.Module):
    """Convolutional Neural Network for plant disease classification"""

    def __init__(self, num_classes, dropout_rate=0.5):
        super(PlantDiseaseModel, self).__init__()
        # Convolutional Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Convolutional Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Convolutional Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Convolutional Block 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Convolutional Block 5
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully Connected Layers
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.global_avg_pool(x)
        x = self.fc_block(x)
        return x

# Early Stopping Utility


class EarlyStopping:
    """Early stopping handler to prevent overfitting"""

    def __init__(self, patience=5, min_delta=0.001, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save(model.state_dict(), self.save_path)
            print(f"[INFO] Model checkpoint saved to {self.save_path}")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("[INFO] Early stopping triggered.")
                return True
        return False

# Data Loading and Preparation Functions


def load_images(directory_root):
    """Load images and their labels from directory structure"""
    image_list, label_list = [], []
    print("[INFO] Loading images...")

    for disease_folder in os.listdir(directory_root):
        disease_folder_path = os.path.join(directory_root, disease_folder)
        if not os.path.isdir(disease_folder_path):
            continue

        for img_name in os.listdir(disease_folder_path):
            if img_name.startswith("."):
                continue
            img_path = os.path.join(disease_folder_path, img_name)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(img_path)
                label_list.append(disease_folder)

    print("[INFO] Image loading completed")
    print(f"Total images: {len(image_list)}")
    return image_list, label_list


def prepare_data(directory_root, image_size=(256, 256), batch_size=32, test_size=0.3, valid_ratio=0.5, random_state=42):
    """Prepare data loaders and label encoder"""
    # Load images and labels
    image_paths, labels = load_images(directory_root)

    # Encode labels as integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Save label encoder for inference
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save class names for reference
    class_names = list(label_encoder.classes_)
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)

    # Train, validation, and test splits
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels_encoded, test_size=test_size, random_state=random_state, stratify=labels_encoded
    )
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=valid_ratio, random_state=random_state, stratify=temp_labels
    )

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(valid_paths)}")
    print(f"Test samples: {len(test_paths)}")

    # Data Transformations
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Save image transformation for inference
    with open('inference_transform.pkl', 'wb') as f:
        pickle.dump(valid_test_transform, f)

    # Create datasets with appropriate transformations
    train_dataset = PlantDiseaseDataset(
        train_paths, train_labels, transform=train_transform)
    valid_dataset = PlantDiseaseDataset(
        valid_paths, valid_labels, transform=valid_test_transform)
    test_dataset = PlantDiseaseDataset(
        test_paths, test_labels, transform=valid_test_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, len(class_names)

# Training and Evaluation Functions


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on validation or test set"""
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(data_loader),
                            desc="Evaluating", total=len(data_loader))
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(
                {"Val Loss": loss.item(), "Accuracy": correct / total * 100})

    val_loss /= len(data_loader)
    accuracy = correct / total * 100
    return val_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None,
                epochs=10, early_stopping=None, device="cpu"):
    """Train the model with optional early stopping and learning rate scheduler"""
    model.to(device)
    train_losses, valid_losses, valid_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{epochs}",
                            total=len(train_loader))

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": loss.item()})

        # Record training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        val_loss, val_accuracy, _, _ = evaluate_model(
            model, valid_loader, criterion, device)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_accuracy)

        # Print epoch summary
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
              f"Val Accuracy = {val_accuracy:.2f}%")

        # Learning rate scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # Early stopping
        if early_stopping and early_stopping(val_loss, model):
            print("[INFO] Early stopping triggered.")
            break

    # Save the learning curves
    save_learning_curves(train_losses, valid_losses, valid_accuracies)

    return train_losses, valid_losses, valid_accuracies


def save_learning_curves(train_losses, valid_losses, valid_accuracies):
    """Save learning curves as a plot"""
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

# Prediction function for inference


def predict_image(model, image_path, transform, device, label_encoder=None):
    """Make prediction on a single image"""
    model.eval()

    # Open and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    predicted_idx = predicted.item()
    confidence = probabilities[0][predicted_idx].item() * 100

    if label_encoder:
        predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    else:
        return predicted_idx, confidence, probabilities[0].cpu().numpy()

# Main function to train the model


def train(data_dir, model_save_path="best_model.pth", batch_size=32,
          epochs=30, learning_rate=0.001, image_size=(256, 256)):
    """Main function to train and save the model and necessary files for deployment"""
    # Prepare data
    train_loader, valid_loader, test_loader, num_classes = prepare_data(
        data_dir, image_size=image_size, batch_size=batch_size
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss function, optimizer and scheduler
    model = PlantDiseaseModel(num_classes=num_classes, dropout_rate=0.5)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    early_stopping = EarlyStopping(
        patience=7, min_delta=0.001, save_path=model_save_path)

    # Print model summary
    print(f"Model created with {num_classes} output classes")

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        early_stopping=early_stopping,
        device=device
    )

    # Load the best model
    model.load_state_dict(torch.load(model_save_path))

    # Evaluate on test set
    print("\n[INFO] Evaluating the model on the test set...")
    test_loss, test_accuracy, predictions, true_labels = evaluate_model(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save model architecture for inference
    dummy_input = torch.randn(1, 3, *image_size).to(device)
    torch.onnx.export(model, dummy_input, "plant_disease_model.onnx")

    # Save model config
    model_config = {
        "image_size": image_size,
        "num_classes": num_classes,
        "model_path": model_save_path,
        "label_encoder_path": "label_encoder.pkl",
        "transform_path": "inference_transform.pkl",
        "class_names_path": "class_names.json"
    }

    with open("model_config.json", "w") as f:
        json.dump(model_config, f)

    print("[INFO] Training completed and all necessary files saved for deployment.")
    return model, model_config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train a plant disease classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--model_path", type=str,
                        default="best_model.pth", help="Path to save the model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_save_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
