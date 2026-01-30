from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get the directory where this setup.py is located
setup_dir = os.path.dirname(os.path.abspath(__file__))
rank_cy_pyx = os.path.join(setup_dir, 'rank_cy.pyx')

setup(ext_modules=cythonize(Extension(
    'utils.rank_cylib.rank_cy',
    sources=[rank_cy_pyx],
    language='c',
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))
