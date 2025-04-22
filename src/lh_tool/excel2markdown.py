import os
import xlrd
import argparse


def get_cell_font_bold(workbook, cell):
    xf_index = cell.xf_index
    xf_style = workbook.xf_list[xf_index]
    font_index = xf_style.font_index
    font = workbook.font_list[font_index]
    return font.bold


def excel2markdown(excel_file, markdown_file, sheet_index=0, style=""):
    markdown_file = os.path.splitext(excel_file)[0] + ".md" if markdown_file is None else markdown_file

    workbook = xlrd.open_workbook(excel_file, formatting_info=True)
    sheet = workbook.sheet_by_index(sheet_index)

    with open(markdown_file, "w") as f:
        f.write("<html>\n")
        f.write('<table align="center">\n')

        merged_cells = dict()
        for (rlow, rhigh, clow, chigh) in sheet.merged_cells:
            merged_cells[(rlow, clow)] = (rhigh - rlow, chigh - clow)

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
            f.write(f"\t<tr{style}>\n")

            for j in range(sheet.ncols):
                cell = sheet.cell(i, j)
                type = cell.ctype
                value = cell.value
                bold = get_cell_font_bold(workbook, cell)
                rowspan = ""
                colspan = ""
                if (i, j) in merged_cells.keys():
                    (rows, cols) = merged_cells[(i, j)]
                    rowspan = f' rowspan="{rows}"'
                    colspan = f' colspan="{cols}"'
                if value != "":
                    if type == 2:
                        precision = precisions[j]
                        value = f"%.{precision}f" % value
                    if bold:
                        f.write(f'\t\t<td{rowspan}{colspan} align="center"><b>{value}</b></td>\n')
                    else:
                        f.write(f'\t\t<td{rowspan}{colspan} align="center">{value}</td>\n')
            f.write("\t</tr>\n")
        f.write("</table>\n")
        f.write("</html>\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input excel file", required=True)
    parser.add_argument("-o", "--output", type=str, help="output markdown file")
    parser.add_argument("-s", "--sheet_index", type=int, default=0, help="sheet index")
    opts = parser.parse_args()
    print(opts)

    excel_file = opts.input
    markdown_file = opts.output
    sheet_index = opts.sheet_index
    style = ' style = "font-size:10px"'
    excel2markdown(excel_file, markdown_file, sheet_index, style)


if __name__ == "__main__":
    main()
