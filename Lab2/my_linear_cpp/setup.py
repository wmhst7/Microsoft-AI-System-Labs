from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='my_linear_cpp',
      ext_modules=[cpp_extension.CppExtension('my_linear_cpp', ['my_linear.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})