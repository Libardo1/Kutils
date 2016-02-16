from setuptools import setup, find_packages

setup(
    name='kutils',
    version='0.1.0',
    url='https://github.com/tdeboissiere/Kutils',
    packages=find_packages(),
    description='Kaggle utilities package',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.16.1",
        "pandas >= 0.17.1",
        "xgboost >= 0.4",
        "matplotlib >= 1.5.1"
    ],
)