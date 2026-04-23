import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    # Your implementation here
    en = lambda x: (x - 4) // 2
    de = lambda x: 2 * x - 4
    en4 = lambda x: en(en(en(en(x))))
    de4 = lambda x: de(de(de(de(x))))
    bottleneck = lambda x: x - 4
    b, h, w, c = x.shape
    h_out = de4(bottleneck(en4(h)))
    w_out = de4(bottleneck(en4(w)))
    return np.ndarray((b, h_out, w_out, num_classes))
