from PIL import Image , ImageDraw 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider  


# Load the image and convert to grayscale
image = "Documents/LinearAlgebra/image.jpg" 
visual_image = Image.open(image).convert("L")

# Convert the image to a matrix
img_matrix = np.array(visual_image)
print(f"Image format in Array {img_matrix}")

# Perform SVD on the image matrix
U, S, VT = np.linalg.svd(img_matrix, full_matrices=False)
print("U shape:", U.shape)
print("S shape:", S.shape)
print("VT shape:", VT.shape)

# Function to reconstruct the image using k singular values
def reconstruct_image(U, S, VT, k):
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    return U_k @ S_k @ VT_k


# Set up the figure for display
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.20)   # Space for slider

# Initial value of k
k0 = 20
img_reconstructed = reconstruct_image(U, S, VT, k0)

# Display the reconstructed image
display_image = ax.imshow(img_reconstructed, cmap='gray')
ax.set_title(f"Reconstructed Image (k = {k0})")
ax.axis("off")

# Create the slider under the image
slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(slider_ax, 'k', 1, len(S), valinit=k0, valstep=1)

# Update the displayed image when slider value changes
def update(val):
    k = int(slider.val)
    new_img = reconstruct_image(U, S, VT, k)
    new_img = np.clip(new_img, 0, 255)
    display_image.set_data(new_img)
    ax.set_title(f"Reconstructed Image (k = {k})")
    fig.canvas.draw_idle()

slider.on_changed(update)

# Show the final interactive plot
plt.show()
