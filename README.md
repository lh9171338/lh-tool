# lh-tool Package

This is a tool package. 

# Updates

 - [x] 2023.06.30: Add image channel conversion between imageio and cv2
 - [x] 2023.06.12: Fix the bug that the ParallelProcess iterator returns results in the wrong order
 - [x] 2023.06.11: Add ParallelProcess and AsyncMultiProcess iterator

# Tools

* image2image
* image2video
* image2gif
* image2pdf
* video2image
* pdf2image
* play_image
* concat_image
* excel2latex
* excel2markdown
* compress_pdf
* startup
* rename_postfix

# Install

* Installing from source
```shell
git clone https://github.com/lh9171338/lh-tool.git

cd lh-tool

py -m build

pip install dist/lh_tool-1.7.1-py3-none-any.whl
```

* Install from the Python Package Index (PyPI)
```shell
pip install lh_tool
```