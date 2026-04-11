import numpy as np

def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    """
    Implement a VGG-style convolutional block.
    """
    # Your implementation here
    return np.zeros((*x.shape[:3], out_channels))