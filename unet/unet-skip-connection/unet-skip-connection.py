import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    # Your implementation here
    _, h_e, w_e, _ = encoder_features.shape
    _, h_d, w_d, _ = decoder_features.shape
    dh = (h_e - h_d) // 2
    dw = (w_e - w_d) // 2
    crop = encoder_features[:, dh:dh + h_d, dw:dw + w_d, :]
    return np.concatenate([crop, decoder_features], axis=3)