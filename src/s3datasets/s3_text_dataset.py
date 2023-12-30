
from .s3_dataset import S3Dataset
from typing import List, Dict


class S3TextDataset(S3Dataset):

    def augment_batch_with_content(self, batch: Dict[str,List[str]]) -> dict:
        keys: List[str] = batch["key"]
        texts: List[str] = [self._load_and_decode_obj(key) for key in keys]
        return { "key": keys, "text": texts }
        
        
