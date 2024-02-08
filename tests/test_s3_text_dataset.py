import sys
import os
import re

path_to_source_modules=os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, path_to_source_modules)

from datasets import Dataset
from s3datasets import S3TextDataset

def test_from_bucket_smoke_test(mock_s3_bucket) -> None:
        bucket_name:str = mock_s3_bucket
        prefix="documents/"
        ds: Dataset = S3TextDataset.from_bucket(bucket_name, prefix=prefix)
        assert ds
        assert re.search('gutenberg',ds[1]["text"], re.IGNORECASE)
