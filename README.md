
# Python Module for Huggingface Datasets from S3

This Python module provides tools to work seamlessly with data stored in Amazon S3 buckets, specifically designed for creating Huggingface `datasets.Dataset` instances. It includes two primary components: `S3Dataset` for creating datasets from S3 objects, and a generator utility for lazily tokenizing text data, facilitating domain adaptation for language models.

## Features

- **S3Dataset**: An adapter for creating lazily-loaded Huggingface datasets from S3 bucket contents. Supports filtering by key prefixes, explicit key lists, or dataset identifiers for flexible dataset creation.
- **TextDS2TokensGenerator**: A utility for generating tokens from dataset items on-the-fly, reducing startup latency and network traffic during the preparation of large datasets for training or fine-tuning language models.

## Installation

Ensure you have Python 3.6+ installed. Install the module and its dependencies via pip:

```sh

pip install s3datasets

# or from source
# pip install "git+https://github.com/stevemadere/s3-datasets.git@v2.1.1"

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

4. **Lazy Tokenization**: Utilize the `TextDS2TokensGenerator` to prepare your data for model training without downloading all of the data before training begins.

```python

    import datasets
    from tokengenerators import TextDS2TokensGenerator

    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(my_hf_dataset,tokenizer, chunk_len=2048, min_stride= 64, max_waste=64)
    training_ds:datasets.IterableDataset = datasets.IterableDataset.from_generator(generator)
```

## Usage

### S3Dataset

`S3Dataset` facilitates the creation of datasets from S3. It supports various modes of selection for bucket objects and allows for lazy loading of data to reduce network traffic and startup delays.  It can also pre-process the content of the S3 objects to interpret them as binary data, text, or json to be automatically decoded into objects.

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

### TextDS2TokensGenerator

The `TextDS2TokensGenerator` is a tool for converting a dataset of text documents into tokenized chunks of fixed length suitable for LLM training with minimal training startup delay. It was designed to ease the process of domain-adapting a large language model (LLM) from a corpus of documents of varying sizes but typically exceeding the context window used for training.

It can be used with Dataset.from_generator() to build a dataset of token sequences of a specified fixed length in a lazy manner, load and tokenizing data just-in-tim as the trainer processes it.

#### Key Features

- **Efficient Tokenization**: Generates token sequences lazily, saving significant memory when working with large datasets.
- **Flexible Document Handling**: Capable of slicing long documents into fixed-length chunks with configurable overlap and waste thresholds, ensuring comprehensive coverage of the text data.
- **Versatile Dataset Compatibility**: Works seamlessly with both `IterableDataset` and regular `Dataset` instances from the Huggingface `datasets` library.
- **Adaptive Stride**: Tokenizes text documents with _"adaptive stride"_ ensuring the longest continuous context possible to maximize real learning.
> e.g. When producing tokenized chunks of 4k tokens, a document that would tokenize to 4k+min_stride+1 tokens total will be tokenized as two chunks of exactly 4k tokens, one anchored at the beginning and the other anchored at the end with whatever overlap is necessary to achieve that. This holds true for any larger documents up to 8k - min_stride.  Once the tokenized length exceeds 8k-min_stride, the number of chunks produced is increased to 3 with substantial stride (overlap) of the chunks.    In this way, the chunks are always 4k long for efficient training and all text is seen by the model with sufficient prefix context (as defined by min_stride)



#### Usage

To utilize the `TextDS2TokensGenerator`, initialize it with your dataset, tokenizer, and configuration for chunk length, stride, and waste. Here is a basic example:

```python
from your_module_name import TextDS2TokensGenerator
from transformers import AutoTokenizer
from datasets import load_dataset

# Load your dataset
dataset = load_dataset('path_to_your_dataset')

# Initialize your tokenizer
tokenizer = AutoTokenizer.from_pretrained('your_preferred_model')

# Create the TextDS2TokensGenerator instance
generator = TextDS2TokensGenerator(
    source_dataset=dataset,
    base_tokenizer=tokenizer,
    text_field_name="text",  # The field in your dataset containing text documents
    chunk_len=4096,          # Desired token sequence length
    min_stride=64,           # Minimum stride between chunks
    max_waste=64,            # Maximum allowed tokens to waste per chunk
    include_all_keys=False   # Whether to include all other keys from the original dataset items
)

# Generate the tokenized dataset
tokenized_dataset = Dataset.from_generator(generator)
```

#### Customization and Advanced Usage

- **Chunk Length (`chunk_len`)**: Adjust this parameter to match the input size expected by your model.
- **Stride (`min_stride`)**: Control the overlap between consecutive chunks to ensure sufficient context to make the training on every token meaningful.
- **Waste (`max_waste`)**: Fine-tune the balance between coverage and efficiency by specifying the maximum number of tokens that can be disregarded (never seen during training) at the end of a document.
- **Inclusion of Original Keys (`include_all_keys`)**: Optionally include all key-value pairs from the original dataset items in the tokenized output, excluding the text itself.  (This is mostly used for debugging and profiling)


### Testing
1. Copy the file _example.env_ to _.env_ and customize its contents
2. pytest

```


## Contributing

Contributions are welcome! Please submit pull requests or open issues to suggest improvements or report bugs.

## License

Apache 2.0

