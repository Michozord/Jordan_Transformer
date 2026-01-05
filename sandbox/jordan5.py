import torch
from torch import nn
import numpy as np
import scipy
from torch.utils.data import DataLoader, TensorDataset

rng = np.random.default_rng(seed=123)

def setup_device():
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
    return device

device = setup_device()

block_types = ["1-1-1-1-1", "1-1-1-2", "1-1-3", "1-2-2", "1-4", "2-3", "5"]
block_type_to_idx = {bt: i for i, bt in enumerate(block_types)}
idx_to_block_type = {i: bt for i, bt in enumerate(block_types)}
superdiagonals = {
    "1-1-1-1-1": [np.array([0., 0., 0., 0.])],
    "1-1-1-2": [np.array([0., 0., 0., 1.]), np.array([0., 0., 1., 0.]), np.array([0., 1., 0., 0.]), np.array([1., 0., 0., 0.])],
    "1-1-3": [np.array([1., 1., 0., 0.]), np.array([0., 1., 1., 0.]), np.array([0., 0., 1., 1.])],
    "1-2-2": [np.array([1., 0., 1., 0.]), np.array([1., 0., 0., 1.]), np.array([0., 1., 0., 1.])],
    "1-4": [np.array([1., 1., 1., 0.]), np.array([0., 1., 1., 1.])],
    "2-3": [np.array([1., 0., 1., 1.]), np.array([1., 1., 0., 1.])],
    "5": [np.array([1., 1., 1., 1. ])],
}

nullity_table = np.array([
    [5, 5, 5, 5],  # "1-1-1-1-1"
    [4, 5, 5, 5],  # "1-1-1-2"
    [3, 4, 5, 5],  # "1-1-3"
    [3, 5, 5, 5],  # "1-2-2"
    [2, 3, 4, 5],  # "1-4"
    [2, 4, 5, 5],  # "2-3"
    [1, 2, 3, 4],  # "5"
], dtype=float)
nullity_table_tensor = torch.tensor(nullity_table, dtype=torch.float32, device=device)


class PerKEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, F_k):
        return self.net(F_k)
    

class KSequenceModel(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=4*dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, Z):
        return self.encoder(Z)
    

class JordanNet(nn.Module):
    def __init__(self, d, num_classes):
        super().__init__()
        self.d = d

        self.per_k = PerKEncoder(in_dim=d*d, out_dim=32)
        self.seq = KSequenceModel(dim=32)

        self.jordan_head = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        # features: (B, d-1, d*d)
        Z = self.per_k(features)        # (B, d-1, 32)
        Z = self.seq(Z)                 # (B, d-1, 32)
        h = Z.mean(dim=1)               # (B, 32)

        logits = self.jordan_head(h)      # raw scores

        return logits 

def kl_loss(logits, target_dist):
    """
    logits: (B, C)
    target_dist: (B, C), sums to 1
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.nn.functional.kl_div(
        log_probs, target_dist, reduction="batchmean"
    )


def tau(eps, c=0.1, eps0=1e-12):
    return c * torch.log(1 + torch.tensor(eps / eps0))

def q_batch(batch_block_types, eps, c=0.1, eps0=1e-12):
    """
    Compute q distributions for a batch of block types.

    batch_block_types: list of strings or tensor of indices (B,)
    eps: float or tensor (B,)
    returns: tensor (B, num_classes) of probabilities
    """
    if isinstance(batch_block_types[0], str):
        batch_indices = torch.tensor([block_type_to_idx[b] for b in batch_block_types],
                                     dtype=torch.long, device=nullity_table_tensor.device)
    else:
        batch_indices = batch_block_types.to(nullity_table_tensor.device)
    
    if abs(eps) <= 1e-12:
        return torch.nn.functional.one_hot(
            batch_indices.cpu(),
            num_classes=len(nullity_table_tensor),
        ).float()

    # Gather the nullities for each batch element: shape (B, d-1)
    batch_nullities = nullity_table_tensor[batch_indices]  # (B, d-1)

    # Expand for broadcasting
    # batch_nullities: (B, 1, d-1)
    # nullity_table_tensor: (1, num_classes, d-1)
    diff = torch.abs(batch_nullities[:, None, :] - nullity_table_tensor[None, :, :])  # (B, num_classes, d-1)
    dists = diff.sum(dim=-1)  # (B, num_classes)

    # Scale by tau and softmax
    tau_val = tau(eps, c=c, eps0=eps0)

    probs = torch.softmax(-dists / tau_val, dim=-1)  # (B, num_classes)

    return probs


def generate_matrix(d, block_type, mode='random', eps=None, lam=1, value_range=None, return_J=False):
    num_possible_blocks = len(superdiagonals[block_type])
    super_diag = superdiagonals[block_type][rng.integers(0, num_possible_blocks)]
    J = lam * np.eye(d) + np.diag(super_diag, k=1)
    
    if eps is not None:
        J += eps * np.random.randn(d, d)

    if value_range is None:
        match mode:
            case "random" | "upper" | "ortho" | "lower":
                value_range = 1
            case "int":
                value_range = 100
            case _:
                raise RuntimeError(f"Mode {mode} is not supported")

    def generate_S():
        while True:
            match mode:
                case "random":
                    S = np.random.randn(d, d) * value_range
                case "int":
                    S = np.random.randint(0, value_range, size=(d, d))
                case "upper":
                    S = np.triu(np.random.randn(d, d)) * value_range
                case "lower":
                    S = np.tril(np.random.randn(d, d)) * value_range
                case "ortho":
                    A = np.random.randn(d, d)
                    Q, _ = np.linalg.qr(A)
                    S = Q
            if abs(np.linalg.cond(S)) < 1e5:
                return S

    S = generate_S()
    if return_J:
        return J, S
    
    X = S @ J @ np.linalg.inv(S)

    # if eps is not None:
    #     X += eps * np.random.randn(d, d)
        
    return X


def per_power_features(X, lam):
    d = X.shape[0]
    N = scipy.linalg.schur(X)[0] - lam * np.eye(d)

    feat_per_k = []
    N_k = N.copy()

    for k in range(1, d):
        feat_per_k.append(N_k.flatten())
        N_k = N_k @ N  # Next power

    return np.stack(feat_per_k)


def generate_training_dataset(matrices_per_class, d=5, mode="random", eps_range=(1e-8, 1e-2), eps=None, lam=1, device="cpu"):

    matrices = []
    labels = []
    features_list = []
    dists_list = []

    for block_type in block_types:
        class_idx = block_type_to_idx[block_type]

        for _ in range(matrices_per_class):
            if eps is None:
                eps = np.exp(np.random.uniform(np.log(eps_range[0]), np.log(eps_range[1])))
            # 1. Generate matrix X
            X = generate_matrix(d, block_type, mode=mode, eps=eps, lam=lam)
            matrices.append(X)
            labels.append(class_idx)

            # 2. Features per power k
            feat_per_k = per_power_features(X, lam)
            features_list.append(feat_per_k)   # (d-1, d+2)

            # 3. Target distribution (KL target)
            dists_list.append(q_batch([block_type], eps=eps).squeeze(0).cpu().numpy())

    # Convert everything to torch tensors
    matrices = torch.tensor(np.stack(matrices), dtype=torch.float32, device=device)
    true_labels = torch.tensor(labels, dtype=torch.long, device=device)
    features = torch.tensor(np.stack(features_list), dtype=torch.float32, device=device)
    dists = torch.tensor(np.stack(dists_list), dtype=torch.float32, device=device)

    return matrices, true_labels, features, dists


def train_jordan_net(
    model,
    features,
    target_dist,
    num_epochs=50,
    batch_size=64,
    lr=1e-3,
    device="cuda",
    patience=3, 
):
    """
    Training loop for JordanNet using custom jordan_loss with validation and early stopping
    """

    # Split train/validation (80/20)
    n_samples = features.size(0)
    idx = torch.randperm(n_samples)
    train_idx = idx[: int(0.8 * n_samples)]
    val_idx = idx[int(0.8 * n_samples) :]

    X_train, X_val = features[train_idx], features[val_idx]
    dist_train, dist_val = target_dist[train_idx], target_dist[val_idx]

    train_dataset = TensorDataset(X_train, dist_train)
    val_dataset = TensorDataset(X_val, dist_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        for batch_features, batch_dist in train_loader:
            batch_features = batch_features.to(device)
            batch_dist = batch_dist.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)

            loss = kl_loss(logits, batch_dist)

            if loss.isnan():
                print("NaN loss encountered, stopping training.")
                return model

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_features.size(0)
        train_loss /= len(train_loader.dataset)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_dist in val_loader:
                batch_features = batch_features.to(device)
                batch_dist = batch_dist.to(device)

                logits = model(batch_features)
                loss = kl_loss(logits, batch_dist)

                val_loss += loss.item() * batch_features.size(0)
        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ===== EARLY STOPPING =====
        if epoch > 4:
            if val_loss < best_val_loss - 1e-6:  # small tolerance
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), "sandbox/model_jordan6.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Best Val Loss: {best_val_loss:.6f}"
                    )
                    model.load_state_dict(torch.load("sandbox/model_jordan6.pth"))
                    break

    return model
