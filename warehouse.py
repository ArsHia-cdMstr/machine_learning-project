# ---------------------------------------------- 1

# height, width, _ = image_hsv.shape
#
# # Create coordinate arrays for X and Y
# X, Y = np.meshgrid(np.arange(width), np.arange(height))
#
# # Scale down the coordinate values
# X_scaled = X / 250
# Y_scaled = Y / 250
#
# # Reshape the scaled coordinate arrays to a 2D array of pixels
# X_pixels = X_scaled.reshape(-1, 1)
# Y_pixels = Y_scaled.reshape(-1, 1)
#
# h_scaled = image_hsv[:, :, 0]
# s_scaled = image_hsv[:, :, 1] / 100
# v_scaled = image_hsv[:, :, 2] / 100
#
# # Reshape the H, S, and V channels to a 2D array of pixels
# h_pixels = h_scaled.reshape(-1, 1)
# s_pixels = s_scaled.reshape(-1, 1)
# v_pixels = v_scaled.reshape(-1, 1)
#
# # Combine the coordinate arrays and channel pixels
# pixels = np.concatenate((h_pixels, s_pixels, v_pixels), axis=1)
#
# # Define the number of clusters for k-means
# num_clusters = 2
#
# # Apply k-means clustering
# kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# kmeans.fit(pixels)
#
# # Get the labels assigned by k-means to each pixel
# labels = kmeans.labels_
#
# # Reshape the labels back to the original image shape
# segmented_image = labels.reshape(height, width)
#
# plt.imshow(image_rgb)
# plt.show()
#
# plt.imshow(segmented_image, cmap='rainbow')
# plt.colorbar()
# plt.show()
#
# plt.imshow(image_rgb)
# plt.show()
#
# # plt.imshow(image_hsv)
# # plt.show()
#
# # Visualize the segmented image
#
# # Visualize the segmented image
# # plt.imshow(segmented_image)
# # plt.colorbar()
# # plt.show()







# --------------------------------------------------- 2


# # Compute the average values for each cluster
# averages = np.zeros((num_clusters, 3))  # Array to store the average values (H, S, V) for each cluster
# counts = np.zeros(num_clusters)  # Array to store the count of pixels in each cluster
#
# for i in range(len(labels)):
#     cluster_label = labels[i]
#     averages[cluster_label] += image_hsv.reshape(-1, 3)[i]
#     counts[cluster_label] += 1
#
# for i in range(num_clusters):
#     averages[i] /= counts[i]
#
# # Print the average values for each cluster
# for i in range(num_clusters):
#     print(f"Cluster {i + 1}: H={averages[i][0]}, S={averages[i][1]}, V={averages[i][2]}")
