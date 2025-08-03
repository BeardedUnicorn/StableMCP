# models/aesthetic.py

import io
from PIL import Image

import torch
import torch.nn as nn
import clip
from huggingface_hub import hf_hub_download

from utils.device import get_device  # your existing device‐selection helper


class MLP(nn.Module):
    """
    Matches the Lightning‐defined MLP from simple_inference.py:
      Linear(768 → 1024) → Dropout(0.2)
      → Linear(1024 → 128) → Dropout(0.2)
      → Linear(128 → 64)  → Dropout(0.1)
      → Linear(64 → 16)
      → Linear(16 → 1)
    No activations between layers.
    """
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticScorer:
    def __init__(self):
        # 1) pick device
        self.device = get_device()

        # 2) load CLIP (ViT‑L/14) and its preprocess fn
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()

        # 3) fetch and load the MLP weights
        ckpt_path = hf_hub_download(
            repo_id="camenduru/improved-aesthetic-predictor",
            filename="ava+logos-l14-linearMSE.pth"
        )
        self.mlp = MLP().to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.mlp.load_state_dict(state)
        self.mlp.eval()

    def score(self, image_bytes: bytes) -> float:
        """
        Given PNG/JPEG bytes, returns a floating‐point aesthetic score.
        """
        # load & preprocess
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 1) CLIP embed
            emb = self.clip_model.encode_image(image_input)
            # 2) normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)
            # 3) MLP → scalar
            score_tensor = self.mlp(emb)

        return float(score_tensor.item())
