[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) lh-tool
===

This is a tool package.

# Updates

 - [x] 2024.09.18: Add pcd2ply module
 - [x] 2023.12.05: Add CI/CD
 - [x] 2023.06.30: Add image channel conversion between imageio and cv2
 - [x] 2023.06.12: Fix the bug that the ParallelProcess iterator returns results in the wrong order
 - [x] 2023.06.11: Add ParallelProcess and AsyncMultiProcess iterator
 - [x] 2023.03.14: Add Timer and TimeConsumption module
 - [x] 2023.03.22: Add file rename module `rename_file`

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
* rename_file
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
* pcd2ply

# Install

* Installing from source
```shell
git clone https://github.com/lh9171338/lh-tool.git

cd lh-tool

python -m build

pip install dist/lh_tool-1.10.2-py3-none-any.whl
```

* Install from the Python Package Index (PyPI)
```shell
# install base
pip install lh_tool

# install with specific dependencies, support: [cv, email, pdf, excel, pcd]
pip install lh_tool[cv]

# install with all dependencies
pip install lh_tool[all]
```
