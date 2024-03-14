[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) lh-tool
===

This is a tool package.

# Updates

 - [x] 2023.12.05: Add CI/CD
 - [x] 2023.06.30: Add image channel conversion between imageio and cv2
 - [x] 2023.06.12: Fix the bug that the ParallelProcess iterator returns results in the wrong order
 - [x] 2023.06.11: Add ParallelProcess and AsyncMultiProcess iterator
 - [x] 2023.03.14: Add Timer and TimeConsumption module

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
* email
* iterotar
    * SingleProcess
    * MultiProcess
    * ParallelProcess
    * MultiThread
    * AsyncProcess
    * AsyncMultiProcess
* timer
* time_consumption

# Install

* Installing from source
```shell
git clone https://github.com/lh9171338/lh-tool.git

cd lh-tool

python -m build

pip install dist/lh_tool-1.9.0-py3-none-any.whl
```

* Install from the Python Package Index (PyPI)
```shell
pip install lh_tool
```
