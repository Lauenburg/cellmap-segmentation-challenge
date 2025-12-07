from cellmap_segmentation_challenge.utils import load_safe_config
import torch

# Load the configuration file
config_path = __file__.replace("process", "train")
config = load_safe_config(config_path)

# Bring the required configurations into the global namespace
batch_size = getattr(config, "batch_size", 8)
input_array_info = getattr(
    config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
)
target_array_info = getattr(config, "target_array_info", input_array_info)
classes = config.classes


# Define the process function, which takes a numpy array as input and returns a numpy array as output
def process_func(x):
    """
    Convert model logits to binary masks.
    
    x: numpy array of raw model outputs (logits)
    Returns: binary numpy array (0 or 1)
    """
    # Apply sigmoid to convert logits to probabilities
    import numpy as np
    probs = 1 / (1 + np.exp(-x))  # sigmoid
    
    # Threshold at 0.5 to get binary masks
    return (probs > 0.5).astype(np.uint8)


if __name__ == "__main__":
    from cellmap_segmentation_challenge import process

    # Call the process function with the configuration file
    process(__file__, overwrite=True)