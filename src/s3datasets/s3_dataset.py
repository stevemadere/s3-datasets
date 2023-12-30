import os
import datasets
import gzip
import boto3
from botocore.client import BaseClient

from typing import List, Union, Dict, Any, cast
import diskcache

cache: diskcache.Cache  = diskcache.Cache('./memo_cache')

VALID_AWS_AUTH_VARS = ['AWS_PROFILE', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_CONFIG_FILE', 'AWS_SHARED_CREDENTIALS_FILE']

class S3Dataset:
    """
    A factory class that creates a Huggingface Dataset connected to an S3 bucket.
    Simply call the static method from_bucket(bucketname,prefix) to get such a Dataset
    """
    bucket_name: str
    s3_client: BaseClient

    def __init__(self,bucket_name: str) -> None:
        # Check that at least one of the VALID_AWS_AUTH_VARS is set in the environment and raise an error if none are
        if not any(var in os.environ for var in VALID_AWS_AUTH_VARS):
            raise ValueError("One of the following env vars must be set to authenticate to AWS: " + ", ".join(VALID_AWS_AUTH_VARS))
        self.bucket_name: str = bucket_name
        self.s3_client= cast(BaseClient, boto3.client('s3')) 

    @classmethod
    def from_bucket(cls, bucket_name: str, prefix: Union[str,None] = None, key_list: Union[List[str],None] = None) -> datasets.Dataset:
        if key_list and prefix:
            raise ValueError("S3Dataset(): caller can provide either an explicit key_list or a prefix but not both")
        s3_dataset: S3Dataset  = cls(bucket_name)
        if not key_list:
            key_list = s3_dataset._get_keys(prefix if prefix else '')
        keys_dataset = datasets.Dataset.from_dict({"key" : key_list})
        #full_dataset = keys_dataset.with_transform(lambda x: { "key": x["key"], "text": s3_text_dataset._load_and_decode_obj(x["key"]) })
        full_dataset = keys_dataset.with_transform(s3_dataset.augment_batch_with_content)
        return full_dataset
        
    def augment_batch_with_content(self, batch: Dict[str,List[str]]) -> dict:
        """ Override this method in subclasses to do the work of converting whatever is in
            the objects in the S3 bucket into the data needed in the Dataset.
            This default implementation is for binary data in a field called "data"
        """
        keys: List[str] = batch["key"]
        blobs: List[bytes] = [self._load_obj(key) for key in keys]
        return { "key": keys, "data": blobs }
        
    
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
        self.s3_client= cast(BaseClient, boto3.client('s3')) 

    def _get_keys(self, prefix: str) -> List[str]:
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        keys: List[str] = [item['Key'] for item in response.get('Contents', [])]
        while response.get('IsTruncated', False):
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, ContinuationToken=response['NextContinuationToken'])
            keys.extend([item['Key'] for item in response.get('Contents', [])])
        return keys

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

