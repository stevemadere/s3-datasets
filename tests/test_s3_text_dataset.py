import sys
import os
import re
import pytest

path_to_source_modules=os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, path_to_source_modules)

from datasets import Dataset
from s3datasets import S3TextDataset

def test_from_bucket_smoke(mock_s3_bucket) -> None:
        bucket_name:str = mock_s3_bucket['bucket_name']
        prefix:str = mock_s3_bucket['document_prefix']
        ds: Dataset = S3TextDataset.from_bucket(bucket_name, prefix=prefix)
        assert ds
        assert re.search('gutenberg',ds[1]["text"], re.IGNORECASE)

def test_from_bucket_with_key_list(mock_s3_bucket) -> None:
        bucket_name:str = mock_s3_bucket['bucket_name']
        all_keys:list[str] = list(mock_s3_bucket['contents'].keys())
        key_list = all_keys[0:3]
        ds3: Dataset = S3TextDataset.from_bucket(bucket_name, key_list=key_list)
        assert ds3
        assert len(list(ds3)) == len(key_list)

def test_from_bucket_with_dataset_id(mock_s3_bucket) -> None:
        bucket_name:str = mock_s3_bucket['bucket_name']
        datasets_dict = mock_s3_bucket['datasets_dict']
        for dataset_id in datasets_dict.keys():
            ds: Dataset = S3TextDataset.from_bucket(bucket_name, dataset_id=dataset_id)
            assert ds
            dataset_members:list[str] = datasets_dict[dataset_id]
            assert len(list(ds)) == len(dataset_members),  (
                f"Dataset with id {dataset_id} expected to have {len(dataset_members)} members, "
                f"but found {len(list(ds))}"
            )

def test_from_bucket_overconstrained(mock_s3_bucket) -> None:
        bucket_name:str = mock_s3_bucket
        prefix="documents"
        dataset_id3="datasets/first_3"
        with pytest.raises(ValueError) as err_info:
            ds_impossible: Dataset = S3TextDataset.from_bucket(bucket_name, prefix=prefix, dataset_id=dataset_id3)
        assert "caller can provide at most one of" in str(err_info.value)

