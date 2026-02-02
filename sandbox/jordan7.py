import torch
from torch import nn
import numpy as np
import scipy
from torch.utils.data import DataLoader, TensorDataset
import random

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

def get_superdiagonal(max_block_size, d):
    longest_run = max_block_size - 1
    n = d - 1
    superdiag = np.zeros(n, dtype=float)
    run_start = random.randint(0, n - longest_run)
    superdiag[run_start:run_start + longest_run] = 1.0

    if run_start - 1 > 0:
        run = 0
        for i in range(run_start - 1):
            if run >= longest_run:
                superdiag[i] = 0.0
                continue
            superdiag[i] = random.choice([0.0, 1.0])
            if superdiag[i] == 1.0:
                run += 1
            else:
                run = 0
    
    if run_start + longest_run + 1 < n:
        run = 0
        for i in range(run_start + longest_run + 1, n):
            if run >= longest_run:
                superdiag[i] = 0.0
                continue
            superdiag[i] = random.choice([0.0, 1.0])
            if superdiag[i] == 1.0:
                run += 1
            else:
                run = 0

    return superdiag


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
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)

    def forward(self, Z, masks=None):
        return self.encoder(Z, src_key_padding_mask=masks.bool() if masks is not None else None)
    

class JordanNet(nn.Module):
    def __init__(self, d, num_classes, encode_dim=32):
        super().__init__()
        self.d = d

        self.per_k = PerKEncoder(in_dim=d*d, out_dim=encode_dim)
        self.seq = KSequenceModel(dim=encode_dim)

        self.jordan_head = nn.Sequential(
            nn.Linear(encode_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, features, masks=None):
        # features: (B, d-1, d*d)
        Z = self.per_k(features)        # (B, d-1, 32)
        Z = self.seq(Z, masks=masks)    # (B, d-1, 32)
        if masks is not None:
            Z = Z.masked_fill(masks.unsqueeze(-1), 0.0)
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

def soft_target(y, eps, d, c=0.1, eps0=1e-8, device_target="cpu"):
    # make y a scalar tensor
    if not torch.is_tensor(y):
        y = torch.tensor(y, device=device_target)
    else:
        y = y.to(device_target)
    y = y.float().view(1)   # shape (1,)

    # class grid
    k = torch.arange(1, d+1, device=device_target, dtype=torch.float32)  # (d,)

    # eps == 0 → one-hot
    if float(eps) <= 1e-8:
        idx = (y.long() - 1).clamp(min=0, max=d-1)  # (1,)
        out = torch.zeros(d, device=device_target)
        out[idx] = 1.0
        return out  # (d,)

    # temperature
    tau = c * torch.log1p(
        torch.tensor(eps, dtype=torch.float32, device=device_target) /
        torch.tensor(eps0, dtype=torch.float32, device=device_target)
    )

    # Gaussian kernel over class index
    logits = -(k - y)**2 / (2 * tau**2)   # (d,)
    return torch.softmax(logits, dim=-1)  # (d,)



def generate_matrix(d, max_block_size, mode='random', eps=None, value_range=None, return_J=False, numpy_float32=False):
    # dtype selection
    dtype = np.float32 if numpy_float32 else np.float64

    super_diag = get_superdiagonal(max_block_size, d).astype(dtype)
    J = np.diag(super_diag, k=1).astype(dtype)
    
    if eps is not None:
        J = J + (eps * np.random.randn(d, d)).astype(dtype)

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
                    S = (np.random.randn(d, d) * value_range).astype(dtype)
                case "int":
                    S = np.random.randint(0, value_range, size=(d, d)).astype(dtype)
                case "upper":
                    S = np.triu(np.random.randn(d, d)).astype(dtype) * value_range
                case "lower":
                    S = np.tril(np.random.randn(d, d)).astype(dtype) * value_range
                case "ortho":
                    A = np.random.randn(d, d).astype(dtype)
                    Q, _ = np.linalg.qr(A)
                    S = Q.astype(dtype)
            if abs(np.linalg.cond(S)) < 1e5:
                return S

    S = generate_S()
    if return_J:
        return J.astype(dtype), S.astype(dtype)
    
    # ensure all operands are same dtype
    S = S.astype(dtype)
    J = J.astype(dtype)
    X = S @ J @ np.linalg.inv(S.astype(dtype))

    # if eps is not None:
    #     X += eps * np.random.randn(d, d)
        
    return X


def per_power_features(X):
    d = X.shape[0]

    feat_per_k = []
    N_k = X.copy()
    r = np.linalg.matrix_rank(X)
    mask = np.ones(d - 1, dtype=bool)
    mask[:r] = 0.0
    if np.all(mask):
        mask[0] = 0.0  # Ensure at least one valid power

    for k in range(1, d):
        feat_per_k.append(N_k.flatten())
        N_k = N_k @ X  # Next power
    
    return np.stack(feat_per_k), mask  # (d-1, d*d), (d-1,)


def generate_training_dataset(matrices_per_class, d=5, mode="random", eps_range=(1e-16, 1e-2), eps=None, no_eps_rate=0.1, device="cpu", numpy_float32=False):

    matrices = []
    labels = []
    features_list = []
    masks_list = []
    dists_list = []

    for max_block_size in range(1, d+1):
        print(f"Generating class with max_block_size={max_block_size}...", end="", flush=True)
        class_idx = max_block_size - 1

        for _ in range(matrices_per_class):
            if eps is None:
                if np.random.uniform(0, 1) < no_eps_rate:
                    eps_l = 0.0
                else:
                    eps_l = np.exp(np.random.uniform(np.log(eps_range[0]), np.log(eps_range[1])))
            else:
                eps_l = eps
            # 1. Generate matrix X
            X = generate_matrix(d, max_block_size, mode=mode, eps=eps_l, numpy_float32=numpy_float32)  # (d, d)
            matrices.append(X)
            labels.append(class_idx)

            # 2. Features per power k
            feat_per_k, mask = per_power_features(X) # (d-1, d*d)
            features_list.append(feat_per_k)   # (d-1, d+2)
            masks_list.append(mask)            # (d-1,)

            # 3. Target distribution (KL target) - compute on CPU only
            y = max_block_size  # pass as int
            dists_list.append(soft_target(y, eps_l, d, device_target="cpu"))
        print("Done.")

    # Convert everything to torch tensors (keep on CPU during generation, only move at end if needed)
    matrices = torch.tensor(np.stack(matrices), dtype=torch.float32, device="cpu")
    true_labels = torch.tensor(labels, dtype=torch.long, device="cpu")
    features = torch.tensor(np.stack(features_list), dtype=torch.float32, device="cpu")
    masks = torch.tensor(np.stack(masks_list), dtype=torch.float32, device="cpu")
    # dists_list contains CPU tensors from `soft_target`
    dists = torch.stack(dists_list)

    # Move to target device only at the very end
    if device != "cpu":
        matrices = matrices.to(device)
        true_labels = true_labels.to(device)
        features = features.to(device)
        masks = masks.to(device)
        dists = dists.to(device)

    return matrices, true_labels, features, masks, dists


def train_jordan_net(
    model,
    features,
    masks,
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
    d = model.d

    # Split train/validation (80/20)
    n_samples = features.size(0)
    idx = torch.randperm(n_samples)
    train_idx = idx[: int(0.8 * n_samples)]
    val_idx = idx[int(0.8 * n_samples) :]

    X_train, X_val = features[train_idx], features[val_idx]
    masks_train, masks_val = masks[train_idx], masks[val_idx]
    dist_train, dist_val = target_dist[train_idx], target_dist[val_idx]

    train_dataset = TensorDataset(X_train, masks_train, dist_train)
    val_dataset = TensorDataset(X_val, masks_val, dist_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    model.load_state_dict(torch.load(f"sandbox/model_jordan7_{d}.pth"))

    for epoch in range(num_epochs):

        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        for batch_features, batch_masks, batch_dist in train_loader:
            batch_features = batch_features.to(device)
            batch_masks = batch_masks.to(device)
            batch_dist = batch_dist.to(device)

            optimizer.zero_grad()
            logits = model(batch_features, batch_masks)
            # logits = model(batch_features, None)
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
            for batch_features, batch_masks, batch_dist in val_loader:
                batch_features = batch_features.to(device)
                batch_masks = batch_masks.to(device)
                batch_dist = batch_dist.to(device)

                logits = model(batch_features, batch_masks)
                # logits = model(batch_features, None)
                loss = kl_loss(logits, batch_dist)

                val_loss += loss.item() * batch_features.size(0)
        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ===== EARLY STOPPING =====
        if True: #epoch > 4:
            if val_loss < best_val_loss - 1e-6:  # small tolerance
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"sandbox/model_jordan7_{d}.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Best Val Loss: {best_val_loss:.6f}"
                    )
                    model.load_state_dict(torch.load(f"sandbox/model_jordan7_{d}.pth"))
                    break

    return model
