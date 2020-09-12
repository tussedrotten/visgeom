from setuptools import setup

install_deps = [
    'numpy',
    'matplotlib'
]

setup(
    name='visgeom',
    version='0.1.0',
    description='A simple visualisation library for 3D poses and cameras in python',
    url='https://github.com/tussedrotten/visgeom',
    author='Trym Vegard Haavardsholm',
    license='BSD-3-Clause',
    packages=['visgeom']
)
