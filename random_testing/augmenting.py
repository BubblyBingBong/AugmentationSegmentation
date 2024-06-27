import cv2
import numpy as np
import random

# THIS IS JUST AUGMENTATION TESTING. THESE AUGMENTATIONS ARE NOT THE ONES USED

file = '../Images/CARLA/CARLA_4.png'
src = cv2.imread(file, 1)

def generateNums():
    return np.random.rand(3, 3)


def generateMatrix():
    mat = np.array([[np.random.normal(1, 0.2), 0, 0],
                    [0, np.random.normal(1, 0.2), 0],
                    [0, 0, np.random.normal(1, 0.2)]])
    # mat = np.array([[1, 0, 0],
    #                 [0, 1, 0],
    #                 [0, 0, 0.2]])
    print(mat)
    return mat
    # return np.identity(3) + np.random.normal(0, 0.3, size=(3, 3))


def generateAugmentation(img):
    A = generateMatrix()
    # img = cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2HLS)
    do_noise = random.randint(0, 1)
    if do_noise == 0:
        print("noise")
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            pixel = img[y][x].T
            img[y][x] = A @ pixel
            if do_noise == 0:
                noise_type = random.randint(1, 20)
                if noise_type == 1:
                    white = random.randint(0, 1)
                    if white == 0:
                        img[y][x] = [0, 0, 0]
                    else:
                        img[y][x] = [255, 255, 255]

    blur_type = random.randint(0, 1)
    if blur_type == 0:
        img = cv2.blur(img, (3, 3))
        print("blurred")

    return img


cv2.imshow('src', src)
src = generateAugmentation(src)
cv2.imshow('augmented src', src)

cv2.waitKey(0)
cv2.destroyAllWindows()
