import os
import cv2
import glob
import tqdm
import argparse


def image2image(image_file_list, postfix, image_size=None):
    for image_file in tqdm.tqdm(image_file_list):
        new_image_file = os.path.splitext(image_file)[0] + f'.{postfix}'
        image = cv2.imread(image_file)
        if image_size is not None:
            image = cv2.resize(image, tuple(image_size))
        cv2.imwrite(new_image_file, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='.', help='path of image files')
    parser.add_argument('-i', '--input_postfix', type=str, default='png', help='original postfix of image filename')
    parser.add_argument('-o', '--output_postfix', type=str, default='jpg', help='desired postfix of image filename')
    parser.add_argument('-s', '--size', type=int, nargs=2, help='desired image size')
    opts = parser.parse_args()
    print(opts)

    image_path = opts.path
    postfix = opts.output_postfix
    image_size = opts.size
    image_file_list = sorted(glob.glob(os.path.join(image_path, f'*.{opts.input_postfix}')))
    image2image(image_file_list, postfix, image_size)


if __name__ == '__main__':
    main()
