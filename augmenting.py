import cv2
import numpy as np
import random

file = 'Images/CARLA/CARLA_4.png'
src = cv2.imread(file, 1)

print(src.shape)
width = src.shape[1]
height = src.shape[0]


def generateNums():
    return np.random.rand(3, 3)


def generateMatrix():
    # mat = np.array([[np.random.normal(1, 0.5), 0, 0],
    #                 [0, np.random.normal(1, 0.5), 0],
    #                 [0, 0, np.random.normal(1, 0.5)]])
    mat = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0.2]])
    print(mat)
    return mat
    # return np.identity(3) + np.random.normal(0, 0.3, size=(3, 3))


def generateImage(count):  # HSL randomization
    A = generateMatrix()
    # img = cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2HLS)
    img = cv2.imread(file, 1)
    for x in range(width):
        for y in range(height):
            # pixel = img[y][x].T
            # img[y][x] = A @ pixel
            img[y][x] = 0.8 * np.array([200, 200, 200]) + 0.2 * np.array(img[y][x])
    # img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    cv2.imshow('img' + str(count), img)


cv2.imshow('src', src)

for i in range(1):
    generateImage(i)

cv2.waitKey(0)
cv2.destroyAllWindows()
