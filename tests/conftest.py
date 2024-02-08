# conftest.py
import os
import pytest
import json
from moto import mock_s3
import datasets
import boto3

my_dirname:str = os.path.dirname(__file__)
fixture_dir=f"{my_dirname}/fixtures"
text_prefix="documents/"

txt_fixture_files = [f for f in os.listdir(fixture_dir) if f.endswith(".txt")]
text_dict = { f'{text_prefix}{fname}':open(f"{fixture_dir}/{fname}").read() for fname in txt_fixture_files }

all_text_keys = list(text_dict.keys())
first_3_keys = all_text_keys[0:3]
last_4_keys = all_text_keys[-4:]
datasets_dict = { 'datasets/first_3': json.dumps(first_3_keys),
                  'datasets/last_4': json.dumps(last_4_keys) }

@pytest.fixture(scope="module")
def mock_s3_bucket():
    with mock_s3():
        s3 = boto3.client('s3')
        bucket_name = 'my-mock-bucket'
        s3.create_bucket(Bucket=bucket_name)
        for key, value in text_dict.items():
            s3.put_object(Bucket=bucket_name, Key=key, Body=value)
        for key, value in datasets_dict.items():
            s3.put_object(Bucket=bucket_name, Key=key, Body=value)
        yield bucket_name

@pytest.fixture(scope="module")
def single_text_item_dataset():
    first_document = next(iter(text_dict.values()))
    yield datasets.Dataset.from_dict({"text": [first_document]})

@pytest.fixture(scope="module")
def multiple_text_item_dataset():
    all_documents = list(text_dict.values())
    yield datasets.Dataset.from_dict({"text": all_documents})

