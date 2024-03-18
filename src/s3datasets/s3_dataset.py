import os
import datasets
import gzip
import boto3
import json
from botocore.client import BaseClient as BotocoreBaseClient
from botocore.exceptions import BotoCoreError

from typing import List, Union, Dict, Any, cast
import diskcache

cache: diskcache.Cache  = diskcache.Cache('./memo_cache')

VALID_AWS_AUTH_VARS = ['AWS_PROFILE', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_CONFIG_FILE', 'AWS_SHARED_CREDENTIALS_FILE']


class S3Dataset:
    """
    This class provides functionality to create lazily-loaded Huggingface datasets
    from objects stored in S3 buckets.

    The constructor is passed criteria to determine which objects in the source S3 Bucket
    are to be used as part of the Dataset.

    Instances of this class can be converted to actual Dataset instances via the methods
    to_keys_dataset() (just the selected keys) and to_full_dataset() (both keys and content).

    The Dataset returned by to_full_dataset() is loaded lazily from a generator to
    reduce memory footprint. i.e.  Initially, only the keys of the selected S3 objects
    are locally stored, it is not until a specific Dataset item is accessed that the
    S3 object assiciated with it is actually fetched from S3.

    The base class implementation of `S3Dataset` produces full Datasets with two columns:
    1. 'key': The key of the object in the S3 bucket.
    2. 'data': The raw binary data of the object.
    This is suitable for images, pdfs, and the like.

    There are subclasses like `S3TextDataset` and `S3JsonDataset` that extend this functionality.
    These subclasses create Datasets with the following columns:
    - `S3TextDataset`: ['key', 'text'], where 'text' is the utf-8 text content of the object.
    - `S3JsonDataset`: ['key', 'obj'], where 'obj' is the deserialized JSON object.

    Attributes:
        bucket_name (str): The name of the S3 bucket connected to this S3Dataset.
        s3_client (BotocoreBaseClient): An instance of a boto3 S3 client to interact with the S3 bucket.

    Example Usage:
        # Create an S3Dataset instance from all objects in the bucket with a specific key prefix.
        dataset = S3Dataset.from_bucket("my_bucket", prefix="images/")

    """
    bucket_name: str
    s3_client: BotocoreBaseClient
    key_list: list[str]

    def __init__(self, bucket_name: str,*,
                    key_list: Union[List[str],None] = None,
                    prefix: Union[str,None] = None,
                    dataset_id: Union[str,None] = None) -> None:
        """
        Constructs an `S3Dataset` instance from specified objects in an Amazon S3 bucket.
        This factory method allows the creation of a Dataset using one of four methods to
        select objects from the bucket. Only one selection method should be used;
        specifying more than one will result in an error.

        The selection methods are as follows:
        1. `key_list`: Directly specify the list of object keys.
        2. `prefix`: Include all objects with keys that start with the given prefix.
        3. `dataset_id`: Use a key that points to a JSON-encoded list of object keys.
        4. Default: If none of the above methods are used, all objects in the bucket are included.

        Args:
            bucket_name (str): The name of the S3 bucket.
            key_list (List[str], optional): A list of object keys for creating the S3Dataset. Defaults to None.
            prefix (str, optional): A prefix to match object keys for S3Dataset creation. Defaults to None.
            dataset_id (str, optional): The key of an S3 object containing a JSON-encoded list of keys
                for S3Dataset creation. Defaults to None.

        Raises:
            ValueError: If more than one selection method (key_list, prefix, dataset_id) is specified.

        Examples:
            s3_dataset = S3Dataset.from_bucket(bucket_name="my_bucket", prefix="images/")
            training_images_ds = s3_dataset.to_full_dataset()
            # This creates an S3Dataset instance including all objects in "my_bucket" with keys starting with "images/".

            s3_dataset = S3TextDataset.from_bucket(bucket_name="my_bucket", dataset_id='datasets/subset1.json.gz')
            training_objects_ds = s3_dataset.to_full_dataset()
            # This creates an S3Dataset instance from the objects whose keys are listed in the S3 object with the key datasets/subset1.json.gz
        """
        # Check that at least one of the VALID_AWS_AUTH_VARS is set in the environment and raise an error if none are
        if not any(var in os.environ for var in VALID_AWS_AUTH_VARS):
            raise ValueError("One of the following env vars must be set to authenticate to AWS: " + ", ".join(VALID_AWS_AUTH_VARS))
        self.bucket_name: str = bucket_name
        self.s3_client= cast(BotocoreBaseClient, boto3.client('s3'))

        num_key_options_provided = sum([prefix is not None, dataset_id is not None, key_list is not None])
        if num_key_options_provided > 1:
            raise ValueError("S3Dataset(): caller can provide at most one of key_list, prefix or dataset_id")
        if key_list:
            self.key_list = list(key_list)
        elif dataset_id:
            self.key_list = self._load_key_list_from_object(dataset_id)
        else:
            effective_prefix = prefix if prefix else ''
            self.key_list = self._get_keys_with_prefix(effective_prefix)

    def keys(self):
        return self.key_list

    def to_keys_dataset(self):
        """ Make a Huggingface dataset containing the keys of this S3Dataset """
        return datasets.Dataset.from_dict({"key" : self.keys()})

    def to_full_dataset(self):
        """
            Generates a Huggingface dataset from the S3Dataset that includes both
            key and content (either 'text' (utf-8 decoded) or 'data' (raw) or 'obj' (json deserialized)) columns
        """
        keys_dataset = self.to_keys_dataset()
        full_dataset = keys_dataset.with_transform(self.augment_batch_with_content)
        return full_dataset

    @classmethod
    def from_bucket(cls, bucket_name: str,
                    # accept any keyword args
                    **kwargs) -> datasets.Dataset:
        """ Deprecated factory method to directly generate a Huggingface dataset
            from a bucket.
            Takes the same arguments as the constructor.

            Instead, you should just call to the constructor to make an
            S3Dataset and then call to_full_dataset() on that object.
            to get the same result.
        """
        s3_dataset: S3Dataset  = cls(bucket_name, **kwargs)
        ds = s3_dataset.to_full_dataset()
        return ds

    def augment_batch_with_content(self, batch: Dict[str,List[str]]) -> dict:
        """ Override this method in subclasses to do the work of converting whatever is in
            the objects in the S3 bucket into the data needed in the Dataset.
            This default implementation is for binary data in a field called "data"
        """
        keys: List[str] = batch["key"]
        blobs: List[bytes] = [self._load_obj(key) for key in keys]
        i =  { "key": keys, "data": blobs }
        #print(f"augmenting and returning {i.keys()}")
        return i

    # custom pickling methods to enable fingerprinting for Dataset with_transform compatibility
    def __getstate__(self) -> Dict[str, Any]:
        state: Dict[str, Any] = self.__dict__.copy()
        # Remove the non-serializable s3_client from the state
        if 's3_client' in state:
            del state['s3_client']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the s3_client
        self.s3_client= cast(BotocoreBaseClient, boto3.client('s3'))

    @cache.memoize()
    def _get_keys_with_prefix(self, prefix: str) -> List[str]:
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        keys: List[str] = [item['Key'] for item in response.get('Contents', [])]
        while response.get('IsTruncated', False):
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, ContinuationToken=response['NextContinuationToken'])
            keys.extend([item['Key'] for item in response.get('Contents', [])])
        return keys

    def _load_key_list_from_object(self, key_of_object_containing_key_list: str) -> List[str]:
        key_list:List[str] = []
        try:
            key_list_json = self._load_and_decode_obj(key_of_object_containing_key_list)
            decoded_key_list:Any = json.loads(key_list_json)
            if isinstance(decoded_key_list, list) and all(isinstance(key,str) for key in decoded_key_list):
                key_list = cast(List[str], decoded_key_list)
            else:
                raise ValueError(f'The JSON-decoded object from a dataset_id must be a list of strings')
        except json.JSONDecodeError as e:
            raise ValueError(f'The object in bucket "{self.bucket_name}" with key "{key_of_object_containing_key_list}" is not valid JSON {e}') from e
        except BotoCoreError as e:
            raise ValueError(f'Unable to load object in bucket "{self.bucket_name}" with key "{key_of_object_containing_key_list}: {e}"') from e
        return key_list


    @cache.memoize()
    def _load_obj(self, key: str) -> bytes:
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        content: bytes = response['Body'].read()
        return content

    def _load_and_decode_obj(self, key: str) -> str:
        obj_data: bytes = self._load_obj(key)
        if key.endswith('.gz'):
            # Decompress the content
            obj_data = gzip.decompress(obj_data)
        return obj_data.decode('utf-8')

