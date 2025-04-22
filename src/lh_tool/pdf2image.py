import os
import glob
import fitz
import argparse
from lh_tool.iterator import SingleProcess, MultiProcess


def pdf2image(pdf_file, image_file, zoom=1):
    pdf = fitz.open(pdf_file)
    trans = fitz.Matrix(zoom, zoom)

    for index in range(pdf.page_count):
        page = pdf[index]
        pm = page.get_pixmap(matrix=trans)
        if pdf.page_count > 1:
            save_file = image_file.replace(".png", f"-{index + 1:06d}.png")
        else:
            save_file = image_file
        pm.save(save_file)
    pdf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input pdf file or path", required=True)
    parser.add_argument("-o", "--output", type=str, help="output image file or path")
    parser.add_argument("-z", "--zoom", type=float, default=1, help="zoom image")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="convert pdf to image files recursively",
    )
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="number of process")
    opts = parser.parse_args()
    print(opts)

    try:
        input_path = opts.input
        output_path = opts.output
        zoom = opts.zoom
        recursive = opts.recursive
        nprocs = opts.nprocs
        if os.path.isfile(input_path):
            nprocs = 1
            input_file_list = [input_path]
            if output_path is not None:
                output_file_list = [output_path]
            else:
                output_file_list = [os.path.splitext(input_path)[0] + ".png"]
        elif os.path.isdir(input_path):
            assert recursive, "Please use -r to convert pdf to image files recursively."
            input_file_list = glob.glob(os.path.join(input_path, "*.pdf"))
            output_file_list = []
            for input_file in input_file_list:
                if output_path is not None:
                    filename = os.path.basename(input_file)
                    filename = os.path.splitext(filename)[0] + ".png"
                    output_file = os.path.join(output_path, filename)
                else:
                    output_file = os.path.splitext(input_file)[0] + ".png"
                output_file_list.append(output_file)
        else:
            raise ValueError("Input path must be a file or directory.")

        if nprocs > 1:
            iterator = MultiProcess(pdf2image, nprocs=nprocs)
        else:
            iterator = SingleProcess(pdf2image)
        iterator.run(input_file_list, output_file_list, zoom=zoom)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
