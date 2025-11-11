from jordanutils import LabelsManager, generate_testset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from jordanutils import *
import pandas as pd


# ---- Define the PyTorch model ----
class SimpleNN(nn.Module):
    def __init__(self, d, width=256, depth=5):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(d * d, width),
            nn.ReLU(),
            *([nn.Linear(width, width), nn.ReLU()] * depth),
            nn.Linear(width, d),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x


# ---- Training + Evaluation function ----
def train_and_test_model(
    train_mode, test_mode, eps=None, verbose=1, epochs=50, width=256, depth=5
):
    # --- Device selection logic ---
    try:
        import torch_directml

        device = torch_directml.device()
        backend = "directml"
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            backend = "cuda"
        else:
            device = torch.device("cpu")
            backend = "cpu"

    if verbose:
        print(f"Using device: {device} (backend: {backend})")

    # --- Dataset setup ---
    d = 5
    dataset_size = 50000
    manager = LabelsManager([0], [1], [2], [3], [4], dataset_size=dataset_size)
    X, y = generate_testset(d, manager, mode=train_mode, eps=eps)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=21
    )

    # --- Convert to tensors ---
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # --- DataLoaders ---
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = SimpleNN(d, width, depth).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Early Stopping setup ---
    best_val_acc = 0.0
    patience = 3
    counter = 0
    best_weights = None

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                pred_labels = preds.argmax(1)
                correct += (pred_labels == yb).sum().item()
                total += yb.size(0)
        train_acc = correct / total

        # --- Validation Loop ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                pred_labels = preds.argmax(1)
                correct += (pred_labels == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        if verbose:
            print(f"Epoch {epoch+1:02d} - Loss: {loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        # --- Early Stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            best_weights = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    # --- Restore best model ---
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # --- Testing ---
    manager.dataset_size = 1000
    X_test, y_test = generate_testset(d, manager, mode=test_mode, eps=eps)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        y_predicted = torch.argmax(outputs, dim=1).cpu().numpy()

    return y_test, y_predicted