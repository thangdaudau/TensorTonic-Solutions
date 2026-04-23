import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    Returns (pool_out, skip_out) as zero arrays with correct shapes.
    """
    # Your implementation here
    batches, h, w, c = x.shape
    skip_shape = np.ndarray((batches, h - 4, w - 4, out_channels))
    pool_shape = np.ndarray((batches, h // 2 - 2, w // 2 - 2, out_channels))
    return pool_shape, skip_shape