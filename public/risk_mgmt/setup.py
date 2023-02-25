from setuptools import setup, find_packages

setup(
    name='risk_mgmt',
    version='1.0',
    description='A module for risk management',
    author='Shanglin Li',
    author_email='sl803@duke.edu',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
    ],
)
