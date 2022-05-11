import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


class GaussianBlur:
    def __init__(self):
        self.kernel = (1/16) * np.array([[1., 2., 1.],
                                         [2., 4., 2.],
                                         [1., 2., 1.]])

    def convolve(self, mat1, mat2):
        return (mat1 * mat2).sum()

    def __call__(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        new_img = np.zeros((rows - 1, cols - 1))
        kern_size = len(self.kernel)

        for i in range(0, rows - (kern_size - 1)):
            for j in range(0, cols - (kern_size - 1)):
                center = [i + 1, j + 1]
                gauss = self.convolve(img[i:kern_size + i, j:kern_size + j], self.kernel)
                new_img[center[0], center[1]] = gauss

        return new_img


if __name__ == "__main__":
    startTime = time.time()
    image = cv2.imread('Images/Lenna.png', cv2.IMREAD_GRAYSCALE)

    gaussian = GaussianBlur()
    new_image = image.copy()

    for i in range(5):
        new_image = gaussian(new_image)

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    plt.title("Original")
    plt.imshow(image)
    plt.gray()
    plt.show()

    plt.title("Gaussian Blurred")
    plt.imshow(new_image)
    plt.gray()
    plt.show()

