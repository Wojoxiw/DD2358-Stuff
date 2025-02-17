'''
Created on 11 feb. 2025

@author: al8032pa
'''
from distutils.core import setup
try: ## need to use this to avoid an import error... for some reason
    from Cython.Build import cythonize
except ImportError:
    # Create a closure for deferred import
    def cythonize(*args, **kwargs):
        from Cython.Build import cythonize
        return cythonize(*args, **kwargs)
import numpy


setup(ext_modules=cythonize("gauss_seidel_cython.pyx", compiler_directives={"language_level": "3"}), include_dirs = [numpy.get_include()])