import numpy as np
import matplotlib.pyplot as plt
from Gaussian import *
import cv2


class HarrisCornerDetection:
    def __init__(self, k=.04, threshold=1000.):
        self.k = k
        self.threshold = threshold

        self.Gx = np.array([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]])

        self.Gy = np.array([[1., 2., 1.],
                            [0., 0., 0.],
                            [-1., -2., -1.]])

        self.G = GaussianBlur()

    @staticmethod
    def convolve(mat1, mat2):
        return (mat1 * mat2).sum()

    def calc_R(self, ni, gx, gy):
        Ixx = gx ** 2
        Ixy = gx * gy
        Iyy = gy ** 2

        Sxx = self.G(Ixx)
        Sxy = self.G(Ixy)
        Syy = self.G(Iyy)

        M = np.array([[Sxx, Sxy],
                      [Sxy, Syy]])

        # R = det(M) - k * (trace ** 2)
        R = Sxx * Syy - np.square(Sxy) - self.k * np.trace(M)

        # Add circles to detected corners
        loc = np.where(R >= self.threshold)
        for pt in zip(*loc[::-1]):
            cv2.circle(ni, pt, 8, (0, 0, 255), 1)

    def __call__(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        new_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kern_size = 3

        Gx = np.zeros((rows - 1, cols - 1))
        Gy = np.zeros((rows - 1, cols - 1))

        for i in range(0, rows - (kern_size - 1)):
            for j in range(0, cols - (kern_size - 1)):
                center = [i + 1, j + 1]
                gx = self.convolve(img[i:kern_size + i, j:kern_size + j], self.Gx)
                gy = self.convolve(img[i:kern_size + i, j:kern_size + j], self.Gy)
                Gx[center[0], center[1]] = gx
                Gy[center[0], center[1]] = gy

        self.calc_R(new_img, Gx, Gy)
        return new_img


if __name__ == "__main__":
    image = cv2.imread('Images/Checkerboard.png')

    harris = HarrisCornerDetection()
    new_image = harris(image)

    plt.title("Original")
    plt.imshow(image)
    plt.gray()
    plt.show()

    plt.title("Corners")
    plt.imshow(new_image)
    plt.show()