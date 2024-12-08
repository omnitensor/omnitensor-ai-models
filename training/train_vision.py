import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_vision(data_dir='data/images', epochs=10, batch_size=32, lr=0.001):
    logger.info("Preparing dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    logger.info("Loading model...")
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Preparing optimizer and loss function...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info("Starting training loop...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    logger.info("Saving model...")
    torch.save(model.state_dict(), 'trained_vision_model.pth')
    logger.info("Training complete and model saved.")

if __name__ == "__main__":
    train_vision()
