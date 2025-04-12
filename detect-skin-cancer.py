from torchvision import transforms, datasets, models
import torch
from torch import nn
import time
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Transformaciones con más data augmentation
    transforms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Carga de datasets
    train_dir = "./skin_cancer_dataset/train/train"
    test_dir = "./skin_cancer_dataset/test/test"
    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Modelo con una capa extra y dropout
    model = models.resnet18(pretrained=True)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    model = model.to('cuda')

    # Configuración
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    num_epochs = 50
    start_time = time.time()

    # Early stopping
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # Entrenamiento
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset) * 100
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc.item())

        print(f"[Train] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

        # Evaluación
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects.double() / len(test_dataset) * 100
        test_loss.append(epoch_loss)
        test_accuracy.append(epoch_acc.item())

        print(f"[Test ] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')  # Guardar el mejor modelo
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🚨 Early stopping activado.")
                break

    total_time = time.time() - start_time
    print(f"\n⏱️ Entrenamiento finalizado en {total_time:.2f} segundos.")

    # Gráficas
    epochs = np.arange(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pérdida por época')
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig("perdida-por-epoca.png")
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accuracy, label='Train Acc')
    plt.plot(epochs, test_accuracy, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Precisión por época')
    plt.legend()
    plt.grid(True)
    plt.savefig("precision-por-epoca.png")
    plt.show()
