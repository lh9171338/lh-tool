import os
import shutil
import argparse
import glob
from lh_tool.image2pdf import image2pdf
from lh_tool.pdf2image import pdf2image


def compress_pdf(input_file, output_file, temp_path, zoom):
    temp_path = os.path.splitext(input_file)[0] if temp_path is None else temp_path
    assert not os.path.exists(temp_path), f"'{temp_path}' exists"

    pdf2image(input_file, None, temp_path, zoom)
    image_file_list = sorted(glob.glob(os.path.join(temp_path, "*.png")))
    image2pdf(image_file_list, output_file)
    shutil.rmtree(temp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input pdf file", required=True)
    parser.add_argument("-o", "--output", type=str, help="output pdf file", required=True)
    parser.add_argument("-t", "--temp_path", type=str, help="temporary path for image files")
    parser.add_argument("-z", "--zoom", type=float, default=1, help="zoom scale")
    opts = parser.parse_args()
    print(opts)

    try:
        input_file = opts.input
        output_file = opts.output
        temp_path = opts.temp_path
        zoom = opts.zoom
        compress_pdf(input_file, output_file, temp_path, zoom)
    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    main()
