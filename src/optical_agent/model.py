"""Conditional variational autoencoder with continuous and categorical heads."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import elite_indices
from .design import condition


class ConditionalVAE(nn.Module):
    def __init__(self, latent=8, width=96):
        super().__init__()
        self.latent = latent
        self.width = width
        self.encoder = nn.Sequential(
            nn.Linear(18, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.mean = nn.Linear(width, latent)
        self.logvar = nn.Linear(width, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent + 3, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.geometry = nn.Linear(width, 9)
        self.glass = nn.Linear(width, 6)

    def encode(self, x, g, c):
        onehot = F.one_hot(g, 3).float().flatten(1)
        h = self.encoder(torch.cat([x, onehot, c], dim=1))
        return self.mean(h), self.logvar(h).clamp(-8, 8)

    def decode(self, z, c):
        h = self.decoder(torch.cat([z, c], dim=1))
        return torch.sigmoid(self.geometry(h)), self.glass(h).reshape(-1, 2, 3)

    def forward(self, x, g, c, stochastic=True):
        mu, lv = self.encode(x, g, c)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if stochastic else mu
        xp, logits = self.decode(z, c)
        return xp, logits, mu, lv

    @torch.no_grad()
    def propose(self, tasks, rng):
        """All randomness comes from an explicit NumPy generator."""
        self.eval()
        z = torch.from_numpy(
            rng.normal(size=(len(tasks), self.latent)).astype(np.float32)
        )
        c = torch.from_numpy(condition(tasks).astype(np.float32))
        x, logits = self.decode(z, c)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        uniform = rng.random((len(tasks), 2, 1))
        g = (uniform > probs.cumsum(axis=-1)).sum(axis=-1).clip(0, 2)
        return x.cpu().numpy().astype(float), g


def loss_fn(output, x, g, beta=0.01):
    xp, logits, mu, lv = output
    continuous = F.mse_loss(xp, x) * 9
    categorical = F.cross_entropy(logits.reshape(-1, 3), g.reshape(-1)) * 0.3
    kl = -0.5 * torch.mean(1 + lv - mu.square() - lv.exp())
    return continuous + categorical + beta * kl


def train(data_path, model_path, epochs=250, seed=2026):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = np.load(data_path)
    train_idx = elite_indices(data, 0)
    val_idx = elite_indices(data, 1)

    def tensors(ids):
        return (
            torch.tensor(data["genes"][ids], dtype=torch.float32),
            torch.tensor(data["glasses"][ids], dtype=torch.long),
            torch.tensor(condition(data["tasks"][ids]), dtype=torch.float32),
        )

    train_set = tensors(train_idx)
    val_set = tensors(val_idx)
    model = ConditionalVAE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    history = []
    best = float("inf")
    best_epoch = 0
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        running = 0.0
        for start in range(0, len(order), 128):
            ids = order[start : start + 128]
            x, g, c = (v[ids] for v in train_set)
            optimizer.zero_grad()
            loss = loss_fn(model(x, g, c), x, g)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            x, g, c = val_set
            val = float(loss_fn(model(x, g, c, stochastic=False), x, g))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running / len(train_idx),
                "validation_loss": val,
            }
        )
        if val < best:
            best = val
            best_epoch = epoch + 1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "latent": model.latent,
                    "width": model.width,
                    "seed": seed,
                    "epoch": best_epoch,
                },
                model_path,
            )
        if (epoch + 1) % 50 == 0:
            print(
                f"epoch {epoch + 1}/{epochs}: train={history[-1]['train_loss']:.4f} val={val:.4f}",
                flush=True,
            )
    report = {
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best,
        "train_examples": len(train_idx),
        "validation_examples": len(val_idx),
        "elite_fraction": 0.2,
        "history": history,
    }
    model_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ConditionalVAE(checkpoint["latent"], checkpoint["width"])
    model.load_state_dict(checkpoint["state_dict"])
    model.training_seed = checkpoint["seed"]
    return model.eval()
