from setuptools import setup, find_packages

setup(
    name='s3datasets',
    version='0.2',
    package_dir={'': 'src'},  # This tells setuptools where to find packages
    packages=find_packages(where='src'),
    install_requires=[
        'datasets',
        'boto3',
        'typing',
        'diskcache'
    ],
)


