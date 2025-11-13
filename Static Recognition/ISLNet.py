import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pathlib

data_dir = pathlib.Path("/kaggle/input/indian-sign-language-islrtc-referred/original_images")

# Check if the directory exists
if not data_dir.exists():
    print(f"Error: The directory '{data_dir}' was not found.")
    print("Please update the 'data_dir' variable to the correct path.")
else:
    # Model parameters
    batch_size = 32
    img_size = 180
    learning_rate = 0.001
    epochs = 15 # You can increase this for better accuracy

    # Set device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. LOAD & PREPARE DATA
    # ---
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), # Converts image to tensor and scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard normalization
    ])

    # Load the entire dataset
    
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # 3. BUILD THE CNN MODEL
    # ---
    class ISLNet(nn.Module):
        def __init__(self, num_classes):
            super(ISLNet, self).__init__()
            # [Image of a Convolutional Neural Network architecture]
        
            self.network = nn.Sequential(
                # Conv Block 1
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 180x180 -> 90x90
        
                # Conv Block 2
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 90x90 -> 45x45
        
                # Conv Block 3
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 45x45 -> 22x22
        
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(64 * 22 * 22, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
                # Softmax is included in the loss function (CrossEntropyLoss)
            )
        
        def forward(self, x):
            return self.network(x)

    model = ISLNet(num_classes).to(device)
    print(model)

    # 4. DEFINE LOSS FUNCTION AND OPTIMIZER
    # ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # 5. TRAINING & VALIDATION LOOP
    # ---
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\nStarting model training...")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation Phase
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print("Training finished!")


    # 6. VISUALIZE TRAINING RESULTS
    # ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history['train_acc'], label='Training Accuracy')
    plt.plot(range(epochs), history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history['train_loss'], label='Training Loss')
    plt.plot(range(epochs), history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # 7. (Optional) SAVE THE MODEL
    # ---
    torch.save(model.state_dict(), 'isl_recognition_model_pytorch.pth')
    print("Model saved as isl_recognition_model_pytorch.pth")
