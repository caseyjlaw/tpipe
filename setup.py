from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

#ext_modules = [Extension("leanpipe_cython", ["leanpipe_cython.pyx"])]
ext_modules = [Extension("qimg_cython", ["qimg_cython.pyx"])]

setup(
#    name = 'leanpipe_cython app',
    name = 'qimg_cython app',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
