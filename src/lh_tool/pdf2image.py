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
            image_file = os.path.join(image_path, f'{index + 1:06d}.png')
        pm.save(image_file)
    pdf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input pdf file', required=True)
    parser.add_argument('-o', '--output', type=str, help='output image file')
    parser.add_argument('-z', '--zoom', type=float, default=1, help='zoom image')
    opts = parser.parse_args()

    opts.output = os.path.splitext(opts.input)[0] + '.png' if opts.output is None else opts.output
    print(opts)

    pdf_file = opts.input
    image_file = opts.output
    image_path = os.path.splitext(image_file)[0]
    zoom = opts.zoom
    pdf2image(pdf_file, image_file, image_path, zoom)


if __name__ == '__main__':
    main()
