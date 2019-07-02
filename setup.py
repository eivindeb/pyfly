from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pyflight',
    version='0.1.0',
    url="https://github.com/eivindeb/pyfly",
    author="Eivind Bøhn",
    author_email="eivind.bohn@gmail.com",
    description="Fixed-Wing Flight Simulator",
    py_modules=['pyflight',],
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    install_requires=requirements,
)