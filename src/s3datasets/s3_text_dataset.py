import datasets
import gzip
import boto3
from botocore.client import BaseClient

from typing import List, Union, Dict, Any
import diskcache

cache: diskcache.Cache  = diskcache.Cache('./memo_cache')

class S3TextDataset:
    """
    Initially, the items are the keys of the objects in an S3 bucket.
    It can be converted to a Dataset containing the actual objects by calling .to_objects() on it.
    """
    bucket_name: str
    s3_client: BaseClient

    def __init__(self,bucket_name: str) -> None:
        self.bucket_name: str = bucket_name
        b3client = boto3.client('s3') # type: BaseClient
        self.s3_client = b3client

    
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
        self.s3_client = boto3.client('s3')


    @staticmethod
    def from_bucket(bucket_name: str, prefix: Union[str,None] = None, key_list: Union[List[str],None] = None) -> datasets.Dataset:
        if key_list and prefix:
            raise ValueError("S3Dataset(): caller can provide either an explicit key_list or a prefix but not both")
        s3_text_dataset = S3TextDataset(bucket_name)
        if not key_list:
            key_list = s3_text_dataset._get_keys(prefix if prefix else '')
        print("key_list is " + key_list.__repr__())
        keys_dataset = datasets.Dataset.from_dict({"key" : key_list})
        print(keys_dataset[1])
        #full_dataset = keys_dataset.with_transform(lambda x: { "key": x["key"], "text": s3_text_dataset._load_and_decode_obj(x["key"]) })
        full_dataset = keys_dataset.with_transform(s3_text_dataset.augment_batch_with_text)
        return full_dataset
        
    def augment_batch_with_text(self, batch: dict) -> dict:
        print("batch = " + batch.__repr__())
        keys = batch["key"]
        texts = [self._load_and_decode_obj(key) for key in keys]
        return { "key": keys, "text": texts }
        
    def _get_keys(self, prefix: str) -> List[str]:
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        keys: List[str] = [item['Key'] for item in response.get('Contents', [])]
        while response.get('IsTruncated', False):
            response = s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, ContinuationToken=response['NextContinuationToken'])
            keys.extend([item['Key'] for item in response.get('Contents', [])])
        return keys

    @cache.memoize()
    def _load_obj(self, key: str) -> bytes:
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        content: bytes = response['Body'].read()
        return content
    
    def _load_and_decode_obj(self, key: str) -> str:
        print("loading " + key.__repr__())
        obj_data: bytes = self._load_obj(key)
        if key.endswith('.gz'):
            # Decompress the content
            obj_data = gzip.decompress(obj_data)
        return obj_data.decode('utf-8')
        
