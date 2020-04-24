from setuptools import setup

setup(
    name='pyfly-fixed-wing',
    version='0.1.2',
    url="https://github.com/eivindeb/pyfly",
    author="Eivind Bøhn",
    author_email="eivind.bohn@gmail.com",
    description="Python Fixed-Wing Flight Simulator",
    packages=["pyfly"],
    package_data={"pyfly": ["x8_param.mat", "pyfly_config.json"]},
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    install_requires=[
        "cycler>=0.10.0",
        "kiwisolver>=1.1.0",
        "matplotlib>=3.1.0",
        "numpy>=1.16.4",
        "pyparsing>=2.4.0",
        "python-dateutil>=2.8.0",
        "scipy>=1.3.0",
        "six>=1.12.0"
  ],
)