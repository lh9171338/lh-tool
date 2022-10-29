import numpy as np
import cv2


def imread(filename, flags=cv2.IMREAD_COLOR):
    """
    功能与 cv2.imread() 一样，解决了中文路径的问题
    :param filename:
    :param flags:
    :return:
    """
    img = cv2.imdecode(np.fromfile(filename, np.uint8), flags)
    return img


def imwrite(filename, img, params=None):
    """
    功能与 cv2.imwrite() 一样，解决了中文路径的问题
    :param filename:
    :param img:
    :param params:
    :return:
    """
    buf = cv2.imencode(filename, img, params)[1]
    buf.tofile(filename)
