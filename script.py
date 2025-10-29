import cv2
import numpy as np
import matplotlib.pyplot as plt

def single_scale_retinex(img, sigma):
    img = img.astype(np.float32) + 1.0

    retinex = np.log10(img + 1e-6) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1e-6)
    return retinex

def multi_scale_retinex(img, sigmas):
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)
    retinex /= len(sigmas)
    return retinex

def simplest_color_balance(img, low_clip, high_clip):
    result = np.zeros_like(img)
    for i in range(img.shape[2]):
        channel = img[:, :, i]
        low_val = np.percentile(channel, low_clip)
        high_val = np.percentile(channel, high_clip)
        channel = np.clip(channel, low_val, high_val)
        channel = (channel - low_val) / (high_val - low_val + 1e-6) * 255
        result[:, :, i] = channel
    return result.astype(np.uint8)

def msr_enhancement(image_path, sigmas=[15, 80, 250], low_clip=1, high_clip=99):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Rasm topilmadi: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB format
    img = img.astype(np.float32) + 1.0

    msr_result = multi_scale_retinex(img, sigmas)

    msr_result = (msr_result - np.min(msr_result)) / (np.max(msr_result) - np.min(msr_result)) * 255
    msr_result = np.clip(msr_result, 0, 255).astype(np.uint8)

    balanced = simplest_color_balance(msr_result, low_clip, high_clip)

    return img.astype(np.uint8), balanced

if __name__ == "__main__":
    input_path = "img1.png"
    original, enhanced = msr_enhancement(input_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced)
    plt.title("Enhanced (MSR) Image")

    plt.show()
    print("Shohjahon Anvarjonov Sherzod o'g'li")


