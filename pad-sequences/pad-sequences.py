import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len == None:
        max_len = max(map(lambda x: len(x), seqs))

    n = len(seqs)
    seqs_padded = np.full((n, max_len), pad_value)
    for i in range(n):
        li = len(seqs[i])
        assign_length = min(li, max_len)
        for j in range(assign_length):
            seqs_padded[i][j] = seqs[i][j]
    return seqs_padded