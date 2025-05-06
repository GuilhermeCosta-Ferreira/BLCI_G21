#test.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils import load_it_data, visualize_img

import timm
from torchvision import transforms
import os
from torchvision.models import resnet50, ResNet50_Weights
# model definition
class Swin_ResNet50_Hybrid(nn.Module):
    def __init__(self, hidden_dim=832, freeze_swin=True, freeze_cnn=True, out_dim=168):
        super().__init__()

        # ResNet50 (CNN branch)
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.cnn_branch = nn.Sequential(*list(resnet.children())[:-6])  # Only go up to and include layer 2
        self.pool_cnn = nn.AdaptiveAvgPool2d((1, 1))  # (B, 512, 1, 1)
        self.output_scale = nn.Parameter(torch.tensor(1.0))  # add in model init
        self.out_dim = out_dim
        if freeze_cnn:
            for p in self.cnn_branch.parameters():
                p.requires_grad = False


        # Swin Tiny (Transformer branch)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)

        if freeze_swin:
            for p in self.swin.parameters():
                p.requires_grad = False
    

        self.decoder = nn.Sequential(
        nn.Linear(832, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, self.out_dim)  # all neurons
    )



    def forward(self, x):
        cnn_feats = self.cnn_branch(x)
        cnn_feats = self.pool_cnn(cnn_feats).flatten(1)

        swin_feats = self.swin(x)

        feats = torch.cat([cnn_feats, swin_feats], dim=1)
        x = self.decoder(feats)
        return self.output_scale * x


# -----------------------------------------------------

def predict_stimulus(
    stimulus_test,
    model_path,
    out_dim=168,
    batch_size=32,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load stimulus
    stimulus_tensor = torch.tensor(stimulus_test).float()

    dataset = TensorDataset(stimulus_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Load model
    model = Swin_ResNet50_Hybrid(out_dim=out_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    all_preds = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            preds = model(batch)
            all_preds.append(preds.cpu())

    predictions = torch.cat(all_preds, dim=0).numpy()  # shape [N, out_dim]
    return predictions

# Example use (if this file is run directly)
if __name__ == "__main__":
    path_to_data = '' ## Insert the folder where the data is, if you download in the same folder as this notebook then leave it blank

    stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)

    preds = predict_stimulus(
        stimulus_test=stimulus_test,
        model_path="hybrid_model_full_neurons.pth",
        out_dim=168
    )
    print("Predictions shape:", preds.shape)
