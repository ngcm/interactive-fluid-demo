from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext as build_pyx
import sys
import numpy
import os

if sys.platform.startswith('darwin'):
    os.environ['CC'] = 'gcc-6'

setup(name='AltSim',
      ext_modules=[Extension('AltSim', ['AltSim.pyx'],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['-O3', '-fopenmp'],
                             extra_link_args=['-O3', '-fopenmp'])],
cmdclass={'build_ext': build_pyx})
