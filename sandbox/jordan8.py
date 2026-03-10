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


class MatrixEncoder(nn.Module):
    def __init__(self, d, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d*d, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, F_k):
        return self.net(F_k)
    
class JordanClassifier(nn.Module):
    def __init__(self, num_classes, in_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, h):
        return self.net(h)
    

class JordanTransformer(nn.Module):
    def __init__(self, dim, num_layers=4, num_heads=8):
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
    def __init__(self, encode_dim=32, num_heads=8):
        super().__init__()
        self.encode_dim = encode_dim

        self.supported_dimensions = set()

        self.encoders = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        self.transformer = JordanTransformer(encode_dim, num_layers=4, num_heads=num_heads)
        self.norm = nn.LayerNorm(encode_dim)

    def add_dimension(self, d):
        if d in self.supported_dimensions:
            raise ValueError(f"Dimension {d} is already supported.")
        self.supported_dimensions.add(d)
        self.encoders[str(d)] = MatrixEncoder(d, out_dim=self.encode_dim)
        self.classifiers[str(d)] = JordanClassifier(num_classes=d, in_dim=self.encode_dim)

    def forward(self, d, features, masks=None):
        # features: (B, d-1, d*d)
        Z = self.encoders[str(d)](features)        # (B, d-1, 32)
        Z = self.norm(Z)                           # Apply LayerNorm here
        Z = self.transformer(Z, masks=masks)       # (B, d-1, 32)
        
        if masks is not None:
            Z = Z.masked_fill(masks.unsqueeze(-1).bool(), 0.0)
        h = Z.mean(dim=1)                          # (B, 32)

        logits = self.classifiers[str(d)](h)       # raw scores
        return logits 

def kl_loss(logits, target_dist):
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.nn.functional.kl_div(
        log_probs, target_dist, reduction="batchmean"
    )

def soft_target(y, eps, d, c=0.1, eps0=1e-8, device_target="cpu"):
    if not torch.is_tensor(y):
        y = torch.tensor(y, device=device_target)
    else:
        y = y.to(device_target)
    y = y.float().view(1)
    k = torch.arange(1, d+1, device=device_target, dtype=torch.float32)

    if float(eps) <= eps0:
        idx = (y.long() - 1).clamp(min=0, max=d-1)
        out = torch.zeros(d, device=device_target)
        out[idx] = 1.0
        return out

    tau = c * torch.log1p(
        torch.tensor(eps, dtype=torch.float32, device=device_target) /
        torch.tensor(eps0, dtype=torch.float32, device=device_target)
    )

    logits = -(k - y)**2 / (2 * tau**2)
    return torch.softmax(logits, dim=-1)

def generate_matrix(d, max_block_size, mode='random', eps=None, value_range=None, return_J=False, numpy_float32=False):
    dtype = np.float32 if numpy_float32 else np.float64
    super_diag = get_superdiagonal(max_block_size, d).astype(dtype)
    J = np.diag(super_diag, k=1).astype(dtype)
    
    if eps is not None:
        J = J + (eps * np.random.randn(d, d)).astype(dtype)

    if value_range is None:
        match mode:
            case "random" | "upper" | "ortho" | "lower": value_range = 1
            case "int": value_range = 100
            case _: raise RuntimeError(f"Mode {mode} is not supported")

    def generate_S():
        while True:
            match mode:
                case "random": S = (np.random.randn(d, d) * value_range).astype(dtype)
                case "int": S = np.random.randint(0, value_range, size=(d, d)).astype(dtype)
                case "upper": S = np.triu(np.random.randn(d, d)).astype(dtype) * value_range
                case "lower": S = np.tril(np.random.randn(d, d)).astype(dtype) * value_range
                case "ortho":
                    A = np.random.randn(d, d).astype(dtype)
                    Q, _ = np.linalg.qr(A)
                    S = Q.astype(dtype)
            if abs(np.linalg.cond(S)) < 1e5:
                return S

    S = generate_S()
    if return_J:
        return J.astype(dtype), S.astype(dtype)
    
    S = S.astype(dtype)
    J = J.astype(dtype)
    X = S @ J @ np.linalg.inv(S.astype(dtype))
    return X


def per_power_features(X):
    d = X.shape[0]
    feat_per_k = []
    N_k = X.copy()
    r = np.linalg.matrix_rank(X)
    mask = np.ones(d - 1, dtype=bool)
    mask[:r+1] = 0.0
    if np.all(mask):
        mask[0] = 0.0

    for k in range(1, d):
        feat_per_k.append(N_k.flatten())
        N_k = N_k @ X
    
    return np.stack(feat_per_k), mask


def generate_training_datasets(matrices_per_class, dimensions=[5], mode="random", eps_range=(1e-16, 1e-2), eps=None, no_eps_rate=0.1, device="cpu", numpy_float32=False):
    dataset = {}
    for d in dimensions:
        matrices, labels, features_list, masks_list, dists_list = [], [], [], [], []
        for max_block_size in range(1, d+1):
            print(f"Generating class with d={d}, max_block_size={max_block_size}...", end="", flush=True)
            class_idx = max_block_size - 1
            for _ in range(matrices_per_class):
                if eps is None:
                    eps_l = 0.0 if np.random.uniform(0, 1) < no_eps_rate else np.exp(np.random.uniform(np.log(eps_range[0]), np.log(eps_range[1])))
                else:
                    eps_l = eps
                X = generate_matrix(d, max_block_size, mode=mode, eps=eps_l, numpy_float32=numpy_float32)
                matrices.append(X)
                labels.append(class_idx)
                feat_per_k, mask = per_power_features(X)
                features_list.append(feat_per_k)
                masks_list.append(mask)
                dists_list.append(soft_target(max_block_size, eps_l, d, device_target="cpu"))
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

        dataset[d] = (matrices, true_labels, features, masks, dists)
    return dataset


def train_jordan_net(
    model,
    training_dataset,
    num_epochs=100,
    batch_size=64,
    lr=1e-4,
    device="cuda",
    patience=3,
    train_transformer=True,
):
    training_dimensions = list(training_dataset.keys())
    filename = f"sandbox/model_jordan8{'_modified' if not train_transformer else ''}.pth"
    history_filename = f"sandbox/history_jordan8{'_modified' if not train_transformer else ''}.csv"
    with open(history_filename, 'w') as f:
        f.write("epoch, train_loss, val_loss, lr")

    if not set(training_dimensions).issubset(model.supported_dimensions):
        raise ValueError(
            f"Model does not support all training dimensions. "
            f"Model supports: {model.supported_dimensions}, "
            f"Training dimensions: {training_dimensions}"
        )

    train_loaders = {}
    val_loaders = {}

    for d in training_dimensions:
        matrices, true_labels, features, masks, target_dist = training_dataset[d]
        n_samples = features.size(0)
        idx = torch.randperm(n_samples)
        train_idx = idx[: int(0.8 * n_samples)]
        val_idx = idx[int(0.8 * n_samples) :]

        X_train, X_val = features[train_idx], features[val_idx]
        masks_train, masks_val = masks[train_idx], masks[val_idx]
        dist_train, dist_val = target_dist[train_idx], target_dist[val_idx]

        train_dataset = TensorDataset(X_train, masks_train, dist_train)
        val_dataset = TensorDataset(X_val, masks_val, dist_val)

        train_loaders[d] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loaders[d] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.train()

    # if train_transformer:
    #     for p in model.transformer.parameters():
    #         p.requires_grad = True
    # else:
    #     for p in model.transformer.parameters():
    #         p.requires_grad = False
    #     for name, p in model.named_parameters():
    #         print(name, p.requires_grad)

    parameters = []
    if train_transformer:
        parameters.extend(list(model.transformer.parameters()))
    for dim in training_dimensions:
        parameters.extend(list(model.encoders[str(dim)].parameters()))
        parameters.extend(list(model.classifiers[str(dim)].parameters()))

    optimizer = torch.optim.AdamW(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        for d in training_dimensions:
            for batch_features, batch_masks, batch_dist in train_loaders[d]:
                batch_features = batch_features.to(device)
                batch_masks = batch_masks.to(device)
                batch_dist = batch_dist.to(device)

                optimizer.zero_grad()
                logits = model(d, batch_features, batch_masks)
                # logits = model(batch_features, None)
                loss = kl_loss(logits, batch_dist)

                if loss.isnan():
                    print("NaN loss encountered, stopping training.")
                    return model

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_features.size(0)
        train_loss /= sum(len(train_loaders[d].dataset) for d in training_dimensions)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for d in training_dimensions:
                for batch_features, batch_masks, batch_dist in val_loaders[d]:
                    batch_features = batch_features.to(device)
                    batch_masks = batch_masks.to(device)
                    batch_dist = batch_dist.to(device)

                    logits = model(d, batch_features, batch_masks)
                    # logits = model(batch_features, None)
                    loss = kl_loss(logits, batch_dist)

                    val_loss += loss.item() * batch_features.size(0)
            val_loss /= sum(len(val_loaders[d].dataset) for d in training_dimensions)

        scheduler.step(val_loss)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        with open(history_filename, '+a') as f:
            f.write(f"\n{epoch+1}, {train_loss:.6f}, {val_loss:.6f}, {optimizer.param_groups[0]['lr']:.2e}")

        # ===== EARLY STOPPING =====
        if True:
            if val_loss < best_val_loss - 1e-6:  # small tolerance
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), filename)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Best Val Loss: {best_val_loss:.6f}"
                    )
                    model.load_state_dict(torch.load(filename))
                    break

    return model