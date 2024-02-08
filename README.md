
# Python Module for Huggingface Datasets from S3

This Python module provides tools to work seamlessly with data stored in Amazon S3 buckets, specifically designed for creating Huggingface `datasets.Dataset` instances. It includes two primary components: `S3Dataset` for creating datasets from S3 objects, and a generator utility for lazily tokenizing text data, facilitating domain adaptation for language models.

## Features

- **S3Dataset**: An adapter for creating lazily-loaded Huggingface datasets from S3 bucket contents. Supports filtering by key prefixes, explicit key lists, or dataset identifiers for flexible dataset creation.
- **Lazy Tokenization**: A utility for generating tokens from dataset items on-the-fly, optimizing memory usage during the preparation of large datasets for training or fine-tuning language models.

## Installation

Ensure you have Python 3.6+ installed. Install the module and its dependencies via pip:

```sh

#for now:
pip install "git+https://github.com/stevemadere/s3-datasets.git@v2.0"

#eventually:

pip install s3-datasets


```

## Quick Start

1. **Configure AWS Credentials**: Set up your AWS credentials via environment variables or AWS configuration files to access your S3 buckets.  e.g. AWS_PROFILE

2. **Create an S3Dataset Instance**: Initialize with a bucket name and selection criteria (prefix, key list, or dataset ID).

```python
from s3datasets import S3TextDataset

# Example: Create dataset from all objects with a specific prefix
s3_dataset: S3TextDataset = S3TextDataset(bucket_name="my_bucket", prefix="my_data/")
```

3. **Convert to Huggingface Dataset**: Use the `to_full_dataset()` method to obtain a dataset instance, ready for use with Huggingface's `datasets` library.

```python
my_hf_dataset:datasets.Dataset = s3_dataset.to_full_dataset()
```

4. **Lazy Tokenization**: Utilize the `TextDS2TokensGenerator` to prepare your data for model training without loading everything into memory.

```python

	import datasets
	from tokengenerators import TextDS2TokensGenerator

    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(my_hf_dataset,tokenizer, chunk_len=2048, min_stride= 64, max_waste=64)
	training_ds:datasets.IterableDataset = datasets.IterableDataset.from_generator(generator)
```

## Usage

### S3Dataset

`S3Dataset` facilitates the creation of datasets from S3. It supports various modes of selection for bucket objects and allows for lazy loading of data to conserve memory.  It can also pre-process the content of the S3 objects to interpret them as binary data, text, or json to be automatically decoded into objects.

#### Initialization

```python
# Direct specification with a list of keys
s3dataset = S3Dataset(bucket_name="my_bucket", key_list=["file1.txt", "file2.txt"])

# Using a dataset ID pointing to a JSON-encoded list of keys
s3dataset = S3Dataset(bucket_name="my_bucket", dataset_id="path/to/key_list.json")

# For either of the examples above, use the subclass S3TextDataset or S3JSONDataset to get utf-8 text or objects respectively

s3textdataset = S3TextDataset(bucket_name="my_bucket", key_list=["textdocs/file1.txt", "textdocs/file2.txt"])
s3jsonataset = S3JSONDataset(bucket_name="my_bucket", key_list=["objects/o1.json", "objects/o2.json"])

```

#### Converting to huggingface dataset

t_dataset:datasets.Dataset = s3textdataset.to_full_dataset()
some_text:str = t_dataset[0]['text']


o_dataset:datasets.Dataset = s3jsondataset.to_full_dataset()
an_object:any = o_dataset[0]['obj']



```


## Contributing

Contributions are welcome! Please submit pull requests or open issues to suggest improvements or report bugs.

## License

Apache 2.0

