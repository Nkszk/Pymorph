from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs


setup(
    package_dir={'':''},
    packages=[
    ],
    ext_modules=[
        Extension('morph',
            sources=[
            'morph.c',
            ],
            include_dirs=[] + get_numpy_include_dirs(),
            library_dirs=[],
            libraries=[],
            extra_compile_args=['-mfpu=neon', '-march=armv7-a', '-O3' ],
            extra_link_args=['modmoe.o', 'lodloe.o']
        )
    ]
)
