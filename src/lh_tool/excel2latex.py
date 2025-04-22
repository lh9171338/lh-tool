import os
import xlrd
import argparse


def get_cell_font_bold(workbook, cell):
    xf_index = cell.xf_index
    xf_style = workbook.xf_list[xf_index]
    font_index = xf_style.font_index
    font = workbook.font_list[font_index]
    return font.bold


def excel2latex(excel_file, text_file, sheet_index=0):
    text_file = os.path.splitext(excel_file)[0] + ".txt" if text_file is None else text_file

    workbook = xlrd.open_workbook(excel_file, formatting_info=True)
    sheet = workbook.sheet_by_index(sheet_index)

    with open(text_file, "w") as f:
        precisions = []
        for j in range(sheet.ncols):
            max_precision = 0
            for i in range(sheet.nrows):
                if sheet.cell_type(i, j) == 2:
                    value = sheet.cell_value(i, j)
                    if value == round(value):
                        precision = 0
                    else:
                        precision = len(str(value).split(".")[-1])
                    max_precision = max(max_precision, precision)
            precisions.append(max_precision)

        for i in range(sheet.nrows):
            for j in range(sheet.ncols):
                cell = sheet.cell(i, j)
                type = cell.ctype
                value = cell.value
                bold = get_cell_font_bold(workbook, cell)
                if type == 2:
                    precision = precisions[j]
                    value = f"%.{precision}f" % value
                if bold:
                    f.write("\\textbf{" + str(value) + "}")
                else:
                    f.write(str(value))
                if j == sheet.ncols - 1:
                    f.write(" \\\\\n")
                else:
                    f.write(" & ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input excel file", required=True)
    parser.add_argument("-o", "--output", type=str, help="output text file")
    parser.add_argument("-s", "--sheet_index", type=int, default=0, help="sheet index")
    opts = parser.parse_args()
    print(opts)

    excel_file = opts.input
    text_file = opts.output
    sheet_index = opts.sheet_index
    excel2latex(excel_file, text_file, sheet_index)


if __name__ == "__main__":
    main()
