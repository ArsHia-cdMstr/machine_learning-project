import pickle
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hsv
import numpy as np
from sklearn.cluster import KMeans

# باز کردن فایل پیکیله شده
with open("images.pkl", "rb") as f:
    images = pickle.load(f)

with open("label.pkl", "rb") as f:
    my_object = pickle.load(f)

# چاپ شیء بازیابی شده
# print(my_object)
print("images : ", images)

print("my_object", my_object)

# نمایش هر عکس
i = 0
for image_rgb in images:
    # print("image rgb: ", image_rgb)

    image_hsv = rgb2hsv(image_rgb)
    i = i + 1
    if i % 2 == 0:
        height, width, _ = image_hsv.shape

        # Create coordinate arrays for X and Y
        X, Y = np.meshgrid(np.arange(width), np.arange(height))

        # Scale down the coordinate values
        X_scaled = X / 500
        Y_scaled = Y / 500

        # Scale down the H, S, and V values
        h_scaled = image_hsv[:, :, 0]
        s_scaled = image_hsv[:, :, 1] / 100
        v_scaled = image_hsv[:, :, 2] / 100

        # Reshape the scaled coordinate arrays to a 2D array of pixels
        X_pixels = X_scaled.reshape(-1, 1)
        Y_pixels = Y_scaled.reshape(-1, 1)

        # Reshape the scaled H, S, and V channels to a 2D array of pixels
        h_pixels = h_scaled.reshape(-1, 1)
        s_pixels = s_scaled.reshape(-1, 1)
        v_pixels = v_scaled.reshape(-1, 1)

        # Combine the coordinate arrays and channel pixels
        pixels = np.concatenate((X_pixels, Y_pixels, h_pixels, h_pixels, s_pixels, v_pixels), axis=1)

        # Define the number of clusters for k-means
        num_clusters = 5

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(pixels)

        # Get the labels assigned by k-means to each pixel
        labels = kmeans.labels_

        # 2 warehouse code

        # Reshape the labels back to the original image shape
        segmented_image = labels.reshape(height, width)

        plt.imshow(image_rgb)
        plt.show()

        # Visualize the segmented image
        plt.imshow(segmented_image, cmap='rainbow')
        plt.colorbar()
        plt.show()

        plt.imshow(image_rgb)
        plt.show()

        # 1 warehouse code
