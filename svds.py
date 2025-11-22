from PIL import Image , ImageDraw 
import numpy as np
import matplotlib.pyplot as plt


#Import image and save the image into a variable
image = "./image.jpg" 
visual_image = Image.open(image).convert("L")

#Transform the image into a matrix 
img_matrix = np.array(visual_image)

print(f"Image format in Array {img_matrix}")

#Perform SVD { SVD Done from NumPy Library}
U, S, VT = np.linalg.svd(img_matrix, full_matrices=False)

print("U shape:", U.shape)
print("S shape:", S.shape)
print("VT shape:", VT.shape)

#Function to return a image with new matrix
def reconstruct_image(U, S, VT, k):
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    return np.dot(U_k, np.dot(S_k, VT_k))


k = 50   # number of singular values to keep
img_reconstructed = reconstruct_image(U, S, VT, k)

# Clip pixel values to [0, 255]
img_reconstructed = np.clip(img_reconstructed, 0, 255)

# ---- Display the result ----
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_matrix, cmap='gray')
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Reconstructed with k = {k}")
plt.imshow(img_reconstructed, cmap='gray')
plt.axis("off")

plt.show()