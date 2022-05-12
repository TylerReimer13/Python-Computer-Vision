import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


class Sobel:
    def __init__(self):
        self.Gx = np.array([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]])

        self.Gy = np.array([[1., 2., 1.],
                            [0., 0., 0.],
                            [-1., -2., -1.]])
    
    @staticmethod
    def convolve(mat1, mat2):
        return (mat1 * mat2).sum()

    def __call__(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        new_img = np.zeros((rows - 1, cols - 1))
        kern_size = 3

        for i in range(0, rows - (kern_size - 1)):
            for j in range(0, cols - (kern_size - 1)):
                center = [i + 1, j + 1]
                gx = self.convolve(img[i:kern_size + i, j:kern_size + j], self.Gx)
                gy = self.convolve(img[i:kern_size + i, j:kern_size + j], self.Gy)
                G = abs(gx) + abs(gy)
                new_img[center[0], center[1]] = G

        return new_img


if __name__ == "__main__":
    startTime = time.time()
    image = cv2.imread('Images/Flower.jpg', cv2.IMREAD_GRAYSCALE)

    sobel = Sobel()
    new_image = sobel(image)

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    plt.title("Original")
    plt.imshow(image)
    plt.gray()
    plt.show()

    plt.title("Edges")
    plt.imshow(new_image)
    plt.gray()
    plt.show()
