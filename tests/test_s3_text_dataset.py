import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from datasets import Dataset
from s3datasets import S3TextDataset

def test_from_bucket() -> None:
    ds: Dataset = S3TextDataset.from_bucket("vast4elephant", "text/arXiv/CL/")
    assert ds
    assert re.search('Abstract\n',ds[1]["text"])
