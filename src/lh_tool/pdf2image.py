import os
import fitz
import argparse


def pdf2image(pdf_file, image_file, image_path, zoom=1):
    pdf = fitz.open(pdf_file)
    trans = fitz.Matrix(zoom, zoom)

    save_to_folder = pdf.page_count > 1 or image_file is None
    if save_to_folder:
        os.makedirs(image_path, exist_ok=True)

    for index in range(pdf.page_count):
        page = pdf[index]
        pm = page.get_pixmap(matrix=trans)
        if save_to_folder:
            image_file = os.path.join(image_path, f"{index + 1:06d}.png")
        pm.save(image_file)
    pdf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="input pdf file", required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="output image file or output path"
    )
    parser.add_argument(
        "-z", "--zoom", type=float, default=1, help="zoom image"
    )
    opts = parser.parse_args()
    print(opts)

    pdf_file = opts.input
    zoom = opts.zoom
    if opts.output is None:
        image_file = os.path.splitext(pdf_file)[0] + ".png"
    else:
        postfix = os.path.splitext(opts.output)[1]
        if len(postfix):
            image_file = opts.output
        else:
            image_file = opts.output + ".png"
    image_path = os.path.splitext(image_file)[0]

    pdf2image(pdf_file, image_file, image_path, zoom)


if __name__ == "__main__":
    main()
