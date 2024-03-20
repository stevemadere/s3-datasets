from setuptools import setup, find_packages

setup(
    name='s3datasets',
    python_requires='>=3.6',
    description = "Lazy-loading HF Datasets sourced from AWS S3 buckets and a chunking text document tokenizer.",
    long_description="This Python module provides tools to work seamlessly with data stored in Amazon S3 buckets, specifically designed for creating Huggingface datasets.Dataset instances. It includes two primary components: S3Dataset for creating datasets from S3 objects, and TextDS2TokensGenerator, a generator utility for lazily tokenizing text data, facilitating domain adaptation for language models.",
    version='2.1.1',
    package_dir={'': 'src'},  # This tells setuptools where to find packages
    packages=find_packages(where='src'),
    install_requires=[
        'datasets',
        'boto3',
        'typing; python_version<"3.5"',
        'diskcache',
        'transformers',
        'torch'
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]

)


