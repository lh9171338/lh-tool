import os
import cv2
import glob
import tqdm
import argparse
from lh_tool.Iterator import SingleProcess, MultiProcess
import lh_tool.imageio as iio


def image2image(input_image_path, output_image_path, input_postfix, output_postfix, image_size=None):
    image_file_list = glob.glob(os.path.join(input_image_path, f'*.{input_postfix}'))
    if len(image_file_list) == 0:
        return
    output_image_path = input_image_path if output_image_path is None else output_image_path
    os.makedirs(output_image_path, exist_ok=True)

    for image_file in tqdm.tqdm(image_file_list):
        filename = os.path.basename(image_file)
        output_image_file = os.path.join(output_image_path, os.path.splitext(filename)[0] + f'.{output_postfix}')
        image = iio.imread(image_file)
        if image_size is not None:
            image = cv2.resize(image, tuple(image_size))
        iio.imwrite(output_image_file, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='.', help='path of input image files')
    parser.add_argument('-o', '--output', type=str, help='path of output image files. The default is the same as the '
                                                         'path of input image files')
    parser.add_argument('-p', '--input_postfix', type=str, default='png', help='original postfix of image filename')
    parser.add_argument('-d', '--output_postfix', type=str, default='jpg', help='desired postfix of image filename')
    parser.add_argument('-s', '--size', type=int, nargs=2, help='desired image size')
    parser.add_argument('-r', '--recursive', action='store_true', help='convert video to images recursively')
    parser.add_argument('-n', '--nprocs', type=int, default=1, help='number of process')
    opts = parser.parse_args()
    print(opts)

    try:
        input_image_path = opts.input
        output_image_path = opts.output
        input_postfix = opts.input_postfix
        output_postfix = opts.output_postfix
        image_size = opts.size
        recursive = opts.recursive
        nprocs = opts.nprocs
        if recursive:
            image_path_list = glob.glob(os.path.join(opts.input, '*/'))
            if nprocs == 1:
                iterator = SingleProcess(image2image)
            else:
                iterator = MultiProcess(image2image, nprocs=nprocs)
            iterator.run(image_path_list, None, input_postfix, output_postfix, image_size)
        else:
            image2image(input_image_path, output_image_path, input_postfix, output_postfix, image_size)

    except AssertionError as e:
        print(e)


if __name__ == '__main__':
    main()
