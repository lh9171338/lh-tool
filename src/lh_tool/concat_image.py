import os
import cv2
import glob
import numpy as np
import argparse
from lh_tool.iterator import SingleProcess, MultiProcess
import imageio.v2 as iio


def concat_image(image_file_list, output_file, padding_size=5, hstack=True, black=True):
    assert len(image_file_list), "The number of image files must greater than 0"

    color = 0 if black else 255
    image = None
    height, width = None, None
    padding = None
    for image_file in image_file_list:
        input_image = iio.imread(image_file)
        h, w = input_image.shape[:2]
        if image is None:
            image = input_image
            height, width = h, w
            padding_shape = list(input_image.shape)
            if hstack:
                padding_shape[1] = padding_size
            else:
                padding_shape[0] = padding_size
            padding = np.full(padding_shape, color, dtype="uint8") * color
        else:
            if hstack:
                input_image = cv2.resize(input_image, (round(w * height / h), height))
                image = np.hstack((image, padding, input_image))
            else:
                input_image = cv2.resize(input_image, (width, round(h * width / w)))
                image = np.vstack((image, padding, input_image))

    iio.imwrite(output_file, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=".",
        help="image file or path of input image files",
    )
    parser.add_argument(
        "-a",
        "--another_input",
        type=str,
        help="another image file or path of input image files",
    )
    parser.add_argument("-o", "--output", type=str, help="output image file or output path")
    parser.add_argument(
        "-p",
        "--postfix",
        type=str,
        default="png",
        help="postfix of image filename",
    )
    parser.add_argument("-c", "--cover", action="store_true", help="cover the input image file")
    parser.add_argument("-s", "--padding_size", type=int, default=5, help="padding size")
    parser.add_argument(
        "-v",
        "--vertical",
        dest="hstack",
        action="store_false",
        help="vertical stack",
    )
    parser.add_argument(
        "-w",
        "--white",
        dest="black",
        action="store_false",
        help="padding with white",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="concatenate images recursively",
    )
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="number of process")
    opts = parser.parse_args()
    print(opts)

    try:
        postfix = opts.postfix
        cover = opts.cover
        padding_size = opts.padding_size
        hstack = opts.hstack
        black = opts.black
        recursive = opts.recursive
        nprocs = opts.nprocs
        if recursive:
            image_file_list = sorted(glob.glob(os.path.join(opts.input, f"*.{postfix}")))
            another_image_file_list = sorted(glob.glob(os.path.join(opts.another_input, f"*.{postfix}")))
            assert len(image_file_list) == len(another_image_file_list), (
                "The number of image files under the two " "paths is not equal"
            )
            output_path = opts.output
            assert cover or output_path is not None, "Output path is 'None'"
            output_path = opts.input if cover else output_path
            os.makedirs(output_path, exist_ok=True)
            output_file_list = [
                os.path.join(output_path, os.path.basename(image_file)) for image_file in image_file_list
            ]
            if nprocs == 1:
                iterator = SingleProcess(concat_image)
            else:
                iterator = MultiProcess(concat_image, nprocs=nprocs)
            iterator.run(
                list(zip(image_file_list, another_image_file_list)),
                output_file_list,
                padding_size=padding_size,
                hstack=hstack,
                black=black,
            )

        else:
            if os.path.isdir(opts.input):
                image_file_list = sorted(glob.glob(os.path.join(opts.input, f"*.{postfix}")))
                output_file = os.path.abspath(opts.input) + f".{postfix}" if opts.output is None else opts.output
                concat_image(image_file_list, output_file, padding_size, hstack, black)
            else:
                image_file = opts.input
                another_image_file = opts.another_input
                image_file_list = [image_file, another_image_file]
                output_file = opts.output
                assert cover or output_file is not None, "Output file is 'None'"

                output_file = image_file if cover else output_file
                concat_image(image_file_list, output_file, padding_size, hstack, black)

    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    main()
