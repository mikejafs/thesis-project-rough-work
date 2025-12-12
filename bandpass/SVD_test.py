import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load a sample grayscale image (8x8 digits)
digits = load_digits()
A = digits.images[0]  # take one digit, e.g., '0'
print("Original matrix shape:", A.shape)

# Step 1: Compute eigendecomposition (via SVD)
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Step 2: Reconstruct using top-k modes
def reconstruct(k):
    """Reconstruct image using top-k singular values/vectors"""
    return U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

# Step 3: Visualize results for different k
ks = [1, 2, 4, 8]
fig, axes = plt.subplots(1, len(ks)+1, figsize=(10, 3))
axes[0].imshow(A, cmap='gray')
axes[0].set_title("Original")

for i, k in enumerate(ks, start=1):
    A_k = reconstruct(k)
    axes[i].imshow(A_k, cmap='gray')
    axes[i].set_title(f"k = {k}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
