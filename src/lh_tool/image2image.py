import os
import cv2
import glob
import argparse
from lh_tool.iterator import SingleProcess, MultiProcess
import imageio.v2 as iio


def image2image(input_file, output_file, image_size=None):
    image = iio.imread(input_file)
    if image_size is not None:
        image = cv2.resize(image, tuple(image_size))
    iio.imwrite(output_file, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=".",
        help="path of input image files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path of output image files. The default is the same as the " "path of input image files",
    )
    parser.add_argument(
        "-p",
        "--input_postfix",
        type=str,
        default="png",
        help="original postfix of image filename",
    )
    parser.add_argument(
        "-d",
        "--output_postfix",
        type=str,
        default="jpg",
        help="desired postfix of image filename",
    )
    parser.add_argument("-s", "--size", type=int, nargs=2, help="desired image size")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="convert video to images recursively",
    )
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="number of process")
    opts = parser.parse_args()
    print(opts)

    try:
        input_path = opts.input
        output_path = opts.output
        input_postfix = opts.input_postfix
        output_postfix = opts.output_postfix
        image_size = None if opts.size is None else tuple(opts.size)
        recursive = opts.recursive
        nprocs = opts.nprocs
        if recursive:
            input_file_list = glob.glob(
                os.path.join(input_path, "**", f"*.{input_postfix}"),
                recursive=True,
            )
        else:
            input_file_list = glob.glob(os.path.join(input_path, f"*.{input_postfix}"))
        output_file_list = []
        for input_file in input_file_list:
            if output_path is not None:
                filename = os.path.basename(input_file)
                filename = os.path.splitext(filename)[0] + f".{output_postfix}"
                output_file = os.path.join(output_path, filename)
            else:
                output_file = os.path.splitext(input_file)[0] + f".{output_postfix}"
            output_file_list.append(output_file)

        if nprocs == 1:
            iterator = SingleProcess(image2image)
        else:
            iterator = MultiProcess(image2image, nprocs=nprocs)
        iterator.run(input_file_list, output_file_list, image_size=image_size)

    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    main()
