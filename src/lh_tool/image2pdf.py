import os
import argparse
import fitz
import glob


def image2pdf(image_file_list, pdf_file):
    if len(image_file_list) == 0:
        return

    doc = fitz.open()
    for image_file in image_file_list:
        imgdoc = fitz.open(image_file)
        pdfbytes = imgdoc.convert_to_pdf()
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insert_pdf(imgpdf)
    doc.save(pdf_file)
    doc.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=".",
        help="input image file or input path",
    )
    parser.add_argument("-o", "--output", type=str, help="output pdf file")
    parser.add_argument(
        "-p",
        "--postfix",
        type=str,
        default="png",
        help="postfix of image filename valid only if " "'input' is a path)",
    )
    opts = parser.parse_args()
    print(opts)

    if os.path.isdir(opts.input):
        image_file_list = sorted(glob.glob(os.path.join(opts.input, f"*.{opts.postfix}")))
        pdf_file = os.path.abspath(opts.input) + ".pdf" if opts.output is None else opts.output
    else:
        image_file_list = [opts.input]
        pdf_file = os.path.splitext(opts.input)[0] + ".pdf" if opts.output is None else opts.output

    image2pdf(image_file_list, pdf_file)


if __name__ == "__main__":
    main()
