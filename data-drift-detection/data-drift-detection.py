import numpy as np

def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    p = np.array(reference_counts)
    q = np.array(production_counts)
    p = p / p.sum()
    q = q / q.sum()
    score = 0.5 * np.fabs(p - q).sum().item()
    drift_detected = score > threshold
    return {"score": score, "drift_detected": drift_detected}