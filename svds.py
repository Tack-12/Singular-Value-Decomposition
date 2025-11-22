from PIL import Image , ImageDraw 
import numpy as np

#Import image and save the image into a variable
image = "./image.jpg" 
visual_image = Image.open(image)

#Transform the image into a matrix 
img_matrix = np.array(visual_image)

print(f"Image format in Array {img_matrix}")


