
from .s3_dataset import S3Dataset
import json
from typing import List, Dict, Any

class S3JsonDataset(S3Dataset):

    def augment_batch_with_content(self, batch: Dict[str,List[str]]) -> dict:
        keys: List[str] = batch["key"]
        json_docs: List[str] = [self._load_and_decode_obj(key) for key in keys]
        objs: List[Any] = [json.loads(json_doc) for json_doc in json_docs]
        return { "key": keys, "obj": objs }
        
        
