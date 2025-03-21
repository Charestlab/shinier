from setuptools import setup, find_packages

setup(
    name='src',
    version='0.1.0',
    description='A Python package for SHINIER TOOLBOX',
    author='Nicolas Dupuis-ROy',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.1',
        'pillow>=9.0.1',
        'matplotlib>=3.9.2'
    ],
    python_requires='>=3.9, <4',
)