import os
import cv2
import glob

import numpy as np
import tqdm
import argparse


def concat_image(image_file_list1, image_file_list2, output_path, padding_size=5, hstack=True, black=True):
    assert len(image_file_list1) == len(image_file_list2), f'The number of images is not equal ' \
                                                           f'{len(image_file_list1)} vs {len(image_file_list2)}'
    os.makedirs(output_path, exist_ok=True)
    color = 0 if black else 255
    for i in tqdm.trange(len(image_file_list1)):
        image_file1 = image_file_list1[i]
        image_file2 = image_file_list2[i]
        image1 = cv2.imread(image_file1)
        image2 = cv2.imread(image_file2)
        shape = list(image1.shape)
        if hstack:
            shape[1] = padding_size
            padding = np.ones(shape) * color
            image = np.hstack((image1, padding, image2))
        else:
            shape[0] = padding_size
            padding = np.ones(shape) * color
            image = np.vstack((image1, padding, image2))
        image_file = os.path.join(output_path, os.path.basename(image_file1))
        cv2.imwrite(image_file, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--input_path1', type=str, default='.', help='path of input image files')
    parser.add_argument('-b', '--input_path2', type=str, help='another path of input image files', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='path of output image files', required=True)
    parser.add_argument('-p', '--padding_size', type=int, default=5, help='padding size')
    parser.add_argument('-v', '--vertical', dest='hsatck', action='store_false', help='vertical stack')
    parser.add_argument('-w', '--white', dest='black', action='store_false', help='padding with white')
    opts = parser.parse_args()
    print(opts)

    input_path1 = opts.input_path1
    input_path2 = opts.input_path2
    output_path = opts.output_path
    hsatck = opts.hsatck
    padding_size = opts.padding_size
    black = opts.black
    image_file_list1 = sorted(glob.glob(os.path.join(input_path1, '*')))
    image_file_list2 = sorted(glob.glob(os.path.join(input_path2, '*')))
    concat_image(image_file_list1, image_file_list2, output_path, padding_size, hsatck, black)


if __name__ == '__main__':
    main()
