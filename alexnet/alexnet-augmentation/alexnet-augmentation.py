import numpy as np

rng = np.random.default_rng()

def random_crop(image: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """Extract a random crop from the image."""
    # YOUR CODE HERE
    h, w, d = image.shape
    i = rng.integers(0, h - crop_size)
    j = rng.integers(0, w - crop_size)
    return image[i : i + crop_size, j : j + crop_size]

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Randomly flip image horizontally."""
    # YOUR CODE HERE
    h, w, d = image.shape
    x = np.array([images[i, ::-1] for i in range(h)]) if rng.random() < 1 - p else image
    return x