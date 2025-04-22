# -*- encoding: utf-8 -*-
"""
@File    :   rename_file.py
@Time    :   2024/03/22 12:40:50
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import glob
import argparse
from lh_tool.iterator import SingleProcess, MultiProcess


def rename(input_file, output_file):
    """rename file"""
    if input_file != output_file:
        os.rename(input_file, output_file)


def main():
    example = """
        example:
            # rename filename
            rename_file -i input -e "lambda x: '{:06d}.{}'.format(int(x.split('.')[0]), x.split('.')[1])"

            # rename file extension
            rename_file -i input -e "lambda x: x.replace('.png', '.jpg')"
    """
    parser = argparse.ArgumentParser(usage=example)
    parser.add_argument("-i", "--input", type=str, default=".", help="path of input files")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path of output files. The default is the same as the " "path of input files",
    )
    parser.add_argument(
        "-e",
        "--expression",
        type=str,
        required=True,
        help="rename lambda expression",
    )
    parser.add_argument("-r", "--recursive", action="store_true", help="rename recursively")
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="number of process")

    opts = parser.parse_args()
    print(opts)

    try:
        input_path = opts.input
        output_path = opts.output
        expression = opts.expression
        expression_func = eval(expression)
        recursive = opts.recursive
        nprocs = opts.nprocs
        if recursive:
            input_file_list = glob.glob(
                os.path.join(input_path, "**", "*"),
                recursive=True,
            )
        else:
            input_file_list = glob.glob(os.path.join(input_path, "*"))
        output_file_list = []
        for input_file in input_file_list:
            filename = os.path.basename(input_file)
            filename = expression_func(filename)
            if output_path is not None:
                output_file = os.path.join(output_path, filename)
            else:
                dirname = os.path.dirname(input_file)
                output_file = os.path.join(dirname, filename)
            output_file_list.append(output_file)

        assert len(output_file_list) == len(
            set(output_file_list)
        ), "output file list is not unique, please check your expression: {}".format(expression)

        process = MultiProcess if nprocs > 1 else SingleProcess
        process(rename, nprocs=nprocs).run(input_file_list, output_file_list)

    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    main()
