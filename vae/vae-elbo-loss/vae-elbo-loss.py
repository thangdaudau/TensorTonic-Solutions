import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    # Your implementation here
    recon = np.sum((x - x_recon) * (x - x_recon), axis=1).mean()
    kl = -0.5 * np.sum(1 + log_var - mu * mu - np.exp(log_var), axis=1).mean()
    total = recon + kl
    return {'total': total, 'recon': recon, 'kl': kl}
