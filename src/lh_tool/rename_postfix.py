import os
import glob
import argparse
from lh_tool.iterator import SingleProcess, MultiProcess


def rename(input_file, output_file):
    os.rename(input_file, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=".", help="path of input files")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path of output files. The default is the same as the " "path of input files",
    )
    parser.add_argument(
        "-p",
        "--input_postfix",
        type=str,
        required=True,
        help="original postfix of image filename",
    )
    parser.add_argument(
        "-d",
        "--output_postfix",
        type=str,
        required=True,
        help="desired postfix of image filename",
    )
    parser.add_argument("-r", "--recursive", action="store_true", help="rename recursively")
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="number of process")
    opts = parser.parse_args()
    print(opts)

    try:
        input_path = opts.input
        output_path = opts.output
        input_postfix = opts.input_postfix
        output_postfix = opts.output_postfix
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
            iterator = SingleProcess(rename)
        else:
            iterator = MultiProcess(rename, nprocs=nprocs)
        iterator.run(input_file_list, output_file_list)

    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    main()
