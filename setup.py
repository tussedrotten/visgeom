from setuptools import setup

setup(
    name='visgeom',
    version='0.1.1',
    description='A simple visualisation library for 3D poses and cameras in python',
    url='https://github.com/tussedrotten/visgeom',
    author='Trym Vegard Haavardsholm',
    license='BSD-3-Clause',
    packages=['visgeom'],
    install_requires=['numpy', 'matplotlib==3.3.1']
)
