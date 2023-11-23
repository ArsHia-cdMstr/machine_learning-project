# import numpy as np
#
# a: np.ndarray = np.arange(50)
# print(type(a))
#
# print(a)
# print(a.shape)
# b: np.ndarray = a.reshape(5, 2, 5)
# print(b)
# print(b.shape)
# b: np.ndarray = a.reshape(-1, 2)
# print(b)
# print(b.shape)

import cv2
from sklearn.cluster import KMeans
import numpy as np

# Load your image
image_bgr = cv2.imread('your_image_path.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert your image to HSV color space
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Reshape your image to a 2D array where each row is a pixel and columns are HSV channels
pixels = image_hsv.reshape((-1, 3))

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42).fit(pixels)

# Reshape the labels from K-means to the shape of the original image
segmented_image = kmeans.labels_.reshape(image_hsv.shape[:2])

# Visualize the segmented image
import matplotlib.pyplot as plt
plt.imshow(segmented_image, cmap='viridis')
plt.show()