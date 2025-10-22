import cv2
import numpy as np

def kmeans_thresholding(image_path, k=2):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pixel_values = gray.reshape((-1, 1)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(gray.shape)

    if k == 2:
        threshold_value = int((centers[1] + centers[0]) / 2)
        _, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        binary_image = segmented_image

    cv2.imshow('Original', gray)
    cv2.imshow('K-means Segmentation', segmented_image)
    cv2.imshow('Threshold Result', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('rasm1.jpg', binary_image)

kmeans_thresholding('rasm.jpg', k=2)

Anvarjonov Shohjahon

