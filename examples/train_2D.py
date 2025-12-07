# This is an example of a training configuration file that trains a 2D U-Net model to predict nuclei and endoplasmic reticulum in the CellMap Segmentation Challenge dataset.

# The configuration file defines the hyperparameters, model, and other configurations required for training the model. The `train` function is then called with the configuration file as an argument to start the training process. The `train` function reads the configuration file, sets up the data loaders, model, optimizer, loss function, and other components, and trains the model for the specified number of epochs.

# The configuration file includes the following components:
# 1. Hyperparameters: learning rate, batch size, input and target array information, epochs, iterations per epoch, random seed, and initial number of features for the model.
# 2. Model: 2D U-Net model with two classes (nuclei and endoplasmic reticulum). (You can also use a 2D ResNet model by uncommenting the relevant lines.)
# 3. Paths: paths for saving logs, model checkpoints, and data split file.
# 4. Spatial transformations: spatial transformations to apply to the training data.

# This configuration file can be used to run training via two different commands:
# 1. `python train_2D.py`: Run the training script directly.
# 2. `csc train train_2D.py`: Run the training script using the `csc train` command-line interface.

# Training progress can be monitored using TensorBoard by running `tensorboard --logdir tensorboard` in the terminal.

# Once the model is trained, you can use the `predict` function to make predictions on new data using the trained model. See the `predict_2D.py` example for more details.

# %% Imports
from upath import UPath
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from cellmap_data.transforms.augment import NaNtoNum, Normalize, Binarize
from cellmap_segmentation_challenge.models import ResNet, UNet_2D

learning_rate = 1e-4
batch_size = 8
warmup_epochs = 10

input_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}

debug_overfit_single_batch = False
iterations_per_epoch = 100
epochs = 150
validation_time_limit = 30
validation_batch_limit = 10
random_seed = 42

classes = ["nuc", "er"]

model_name = "2d_unet"
model_to_load = "2d_unet"
model = UNet_2D(1, len(classes), trilinear=True)

load_model = "latest"

logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"


spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

max_grad_norm = 0.5
gradient_accumulation_steps = 2

filter_by_scale: False      
weighted_sampler: False     
weight_loss: True           
skip_all_zero_targets: True 
force_all_classes: "train"  

# Optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_kwargs = {"T_max": epochs, "eta_min": 1e-6}

train_raw_value_transforms = T.Compose([
    T.ToDtype(torch.float, scale=True),
    Normalize(),
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
])

val_raw_value_transforms = T.Compose([
    T.ToDtype(torch.float, scale=True),
    Normalize(),
    NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
])

target_value_transforms = T.Compose([
    T.Lambda(lambda x: (x > 0).float()),  # First convert IDs to binary
    T.ToDtype(torch.float, scale=True),
])


class SoftDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1.0, pos_weight=None):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)

        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        if isinstance(pos_weight, torch.Tensor):
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype).clamp(0.0, 1.0)

        pos_weight = getattr(self, "pos_weight", None)
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )

        probs = torch.sigmoid(logits)
        dims = tuple(range(2, probs.dim()))
        intersection = (probs * targets).sum(dim=dims, keepdim=True)
        denom = probs.sum(dim=dims, keepdim=True) + targets.sum(dim=dims, keepdim=True)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_expanded = dice.expand_as(logits)

        loss = self.bce_weight * bce + self.dice_weight * dice_expanded
        return loss


criterion = SoftDiceBCELoss
criterion_kwargs = {
    "dice_weight": 0.6,
    "bce_weight": 0.4,
    "smooth": 1e-3,
    "pos_weight": [1, 2]
}


if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)