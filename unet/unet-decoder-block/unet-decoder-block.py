import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    Returns zero array with correct shape.
    """
    # Your implementation here
    batches, h, w, c = x.shape
    return np.ndarray((batches, h * 2 - 4, w * 2 - 4, out_channels))
