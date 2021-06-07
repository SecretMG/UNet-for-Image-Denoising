import copy
import os
import pdb
import cv2 as cv
import numpy as np
from tqdm import tqdm


target_size = (160, 160)

def get_GT():
    dir_input = '../dataset/helen_1'
    dir_output = '../dataset/UNet/GT'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    for root, dirs, files in os.walk(dir_input):
        for file in tqdm(files):
            p = os.path.join(root, file)
            img = cv.imread(p)
            target = img[
                0:target_size[0], 0:target_size[1]
            ]
            if target.shape[0] == target_size[0] == target.shape[1]:
                target_p = os.path.join(dir_output, file)
                cv.imwrite(target_p, target)


def get_Noised(flag_gauss, flag_salt_pepper, sigma=30, num_peppers=2000):
    if flag_gauss:
        pass
    if flag_salt_pepper:
        pass
    dir_input = '../dataset/UNet/GT'
    dir_output = '../dataset/UNet/input'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    for root, dirs, files in os.walk(dir_input):
        for file in tqdm(files):
            p_in = os.path.join(root, file)
            p_out = os.path.join(dir_output, file)
            img = (cv.imread(p_in) / 255).astype(float)
            if flag_gauss:
                img += np.random.normal(0, sigma/255, img.shape)
                img = np.minimum(1, img)
                img = np.maximum(0, img)
            if flag_salt_pepper:
                for idx in range(num_peppers):
                    x, y = np.random.randint(0, target_size[0]), np.random.randint(0, target_size[0])
                    if idx % 2:
                        img[x, y] = 0
                    else:
                        img[x, y] = 1
            cv.imwrite(p_out, img*255)

def main():
    get_GT()
    get_Noised(flag_gauss=True, flag_salt_pepper=True, sigma=30, num_peppers=2000)


if __name__ == '__main__':
    main()
