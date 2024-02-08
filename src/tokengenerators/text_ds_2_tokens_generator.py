import torch
from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset, IterableDataset
from typing import NewType, Dict, List, Any, Iterator, cast


# Why do I need a checksum of my module source code? you may ask.
#  Well, let me tell you a story of hair-pulling insanity during debugging when
#  huggingface Dataset decides to cache the implementation of my class
#  *in the freaking dataset cache*  and uses old versions of the code (some but not all of it!)
#  while executing my dataaset generator even as I'm debugging and modifying my source code. 
#  To prevent this insanity-inducing infuriating behavior, I need to make sure the
#  'signature' of my Dataset (which depends on the generator) changes if my source code
#  changes.  Thus, I need to include MODULE_CHECKSUM somewhere in the pickled rendering
#  of the generator to ensure that if the code changes at all, the Dataset's fingerprint
#  will change and huggingface Dataset won't just use cached data from previous runs,
#  bypassing all of my recently introduced debugging probes. (print, assert, etc.)

import hashlib
import os

# Global variable for checksum
MODULE_CHECKSUM = None

def compute_checksum(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Compute checksum using __file__ to get the current module's path
module_path = os.path.abspath(__file__)
# F-You huggingface Dataset caching!
MODULE_CHECKSUM = compute_checksum(module_path)


DSItem = NewType('DSItem',Dict[str,Any])

#TokenizerListResult = NewType('TokenizerListResult', Dict[str,List])
#TokenizerTensorResult = NewType('TokenizerTensorResult', dict[str,torch.Tensor])

def dataset_to_iterable(dataset: Dataset) -> Iterator[DSItem]:
    #print(f"class of dataset being converted to iterable is {dataset.__class__}")
    for i in range(len(dataset)):
        yield cast(DSItem,dataset[i])

class TextDS2TokensGenerator:
    """
        Produces generators that can be used with Dataset.from_generator()
        to tokenize a series of text Dataset items.
        Instantiate a member of this class with an underlying dataset (which can be an IterableDataset or a Dataset)
        containing text documents to be tokenized.
        This generator generates token sequences of a specified fixed length lazily,
        thus saving memory when your dataset of text is absolutely huge.
        It works particularly well in combination with S3TextDataset which lazily loads text documents from an S3 bucket.

        example usage:

        s3_dataset = S3TextDataset(my_bucket_name, key_prefix)
        ds_dict = s3_dataset.train_test_split(test=0.05)
        ds_generator = TextDS2TokensGenerator(ds_dict["train"], my_tokenizer, text_field_name = "text", chunk_len = 4096, min_stride = 64, max_waste = 64)
        training_dataset = Dataset.from_generator(ds_generator)

        
    """
    construction_dataset: Dataset|IterableDataset
    source_dataset: Iterator[DSItem]
    base_tokenizer: PreTrainedTokenizerFast
    text_field_name: str 
    chunk_len: int
    min_stride: int
    max_waste: int
    include_all_keys: bool
    _current_item_generator:Iterator[DSItem]|None

    def __init__(self,source_dataset: Dataset|IterableDataset, base_tokenizer: PreTrainedTokenizerFast, text_field_name: str = "text", chunk_len: int = 4096, min_stride: int = 64, max_waste: int = 64, verbose: bool = False, include_all_keys: bool = False) -> None:
        assert min_stride < chunk_len
        self.construction_dataset = source_dataset
        self.source_dataset = cast(Iterator[DSItem],self.construction_dataset) if isinstance(self.construction_dataset, IterableDataset) else dataset_to_iterable(self.construction_dataset)
        #self.source_dataset = cast(Iterator[DSItem],self.construction_dataset)
        self.base_tokenizer = base_tokenizer
        self.text_field_name = text_field_name
        self.chunk_len = chunk_len
        self.min_stride = min_stride
        self.max_waste = max_waste
        self.verbose = verbose
        self.include_all_keys = include_all_keys
        self._current_item_generator = None


    def yield_tokenized_chunks_from_text_item(self, text_item: DSItem) -> Iterator[DSItem]:
        text: str = text_item[self.text_field_name]
        #print(f"text_item keys: {text_item.keys()}")
              # {text_item['key']}")
        tokens: BatchEncoding = self.base_tokenizer(text, max_length = 1024*1024*1024, return_length = True, truncation=True)
        #print(f"tokens: {tokens.__repr__()}")
        # not sure why pyright is complaining about the following, ignore it
        num_tokens: int = tokens["length"][0] # type: ignore
        assert isinstance(num_tokens, int)
        slices: List[slice] = []
        # check if the document is small enough to generate just one chunk
        if num_tokens <= self.chunk_len + self.max_waste:
            slices = [slice(0,num_tokens)]
            if num_tokens < self.chunk_len:
                tokens = self.base_tokenizer(text, padding="max_length", max_length= self.chunk_len, return_length = True)
        else:
            # The document is too long and must be sliced into multiple chunks
            optimal_step_size: float = TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens, self.chunk_len, self.min_stride, self.max_waste)
            slices = TextDS2TokensGenerator.calculate_slices(num_tokens, self.chunk_len, optimal_step_size)
        slice_index = 0
        for chunk_slice in slices:
            generated_item:DSItem = DSItem({"input_ids":None})
            for k,v in tokens.items():
                if k == "length":
                    continue
                values_slice = v[chunk_slice].copy()
                t = torch.tensor(values_slice)
                generated_item[k]=t
                if k == "input_ids":
                    generated_item["labels"] = torch.tensor(values_slice.copy())
            if self.include_all_keys:
                # Also include all members of the original text dataset item except for the text
                for k,v in text_item.items():
                    #print(f"in yield_tokenized_chunks_from_text_item: considering duplication of key '{k}'")
                    if not ( k in generated_item or k == self.text_field_name):
                        #print(f"in yield_tokenized_chunks_from_text_item: duplicating key '{k}' with value '{v}'")
                        generated_item[k] = v
            generated_item["slice_index"] = slice_index
            if self.verbose:
                print(f"yielding generated_item {generated_item.__repr__()}")
            yield generated_item
            slice_index+=1

    # Need a callable form to make huggingface Dataset.from_generator(me) happy
    def __call__(self):
        iter = self.__iter__()
        for x  in iter:
            yield(x)

    def __iter__(self):
        self._current_item_generator=None
        return self

    def _get_next_source_item(self):
        #print(f"source_dataset: {self.source_dataset.__repr__()}")
        source_item:Any = next(self.source_dataset)
        current_source_item = cast(DSItem,source_item)
        #print(f"current_source_item keys: {current_source_item.keys()}")
        assert isinstance(current_source_item[self.text_field_name], str)
        self._current_item_generator = self.yield_tokenized_chunks_from_text_item(current_source_item)

    def __next__(self) -> DSItem:
        while True:
            if not self._current_item_generator:
                self._get_next_source_item() # automatically raises StopIteration if exhausted
            try:
                item = next(cast(Iterator[DSItem],self._current_item_generator))
                return item
            except StopIteration:
                self._current_item_generator = None

    @staticmethod
    def  calculate_optimal_step_size(num_tokens: int, chunk_len: int, min_stride: int, max_waste: int) -> float:
        assert num_tokens > chunk_len + max_waste

        # there is always at least one chunk, take that off and see what's left to cover

        remaining_tokens = num_tokens - chunk_len
        remaining_partitions: int = (((remaining_tokens-max_waste) -1)  // (chunk_len - min_stride) ) + 1

        #print(f"remaining_partitions = {remaining_partitions}")

        # After the first chunk is laid down, determine what is left to cover by remaining chunks and allocate it
        ideal_partition_len: float = remaining_tokens / remaining_partitions
        #print(f"ideal_partition_len = {ideal_partition_len}")
        optimal_stride: float = chunk_len - ideal_partition_len
        #print(f"optimal_stride = {optimal_stride}")
        if optimal_stride < min_stride:
            optimal_stride = 1.0*min_stride
        optimal_step_size:float =  chunk_len - optimal_stride
        return optimal_step_size
        
    @staticmethod
    def calculate_slices(num_tokens: int, chunk_len: int, optimal_step_size: float) -> List[slice]:
        slices: List[slice] = []
        offset: float = 0.0
        last_possible_offset:float = (num_tokens - chunk_len)+0.4
        while offset <= last_possible_offset:
            slice_start: int = int(offset+0.5)
            slice_end: int = slice_start + chunk_len
            # floating point rounding error accumulation paranoia:  ensure we never accidentally extend beyond the end
            if (slice_end > num_tokens):
                slice_end = num_tokens
                slice_start = slice_end - chunk_len
            slices.append(slice(slice_start,slice_end))
            offset += optimal_step_size
        return slices

    # custom pickling methods to enable fingerprinting for Dataset.with_transform() compatibility 
    # No pickling/unpickling will actually ever be desired, just a signature to detect changes
    @staticmethod
    def _reconstruct(cls, args, checksum):
        obj = cls(*args)
        return obj

    def __reduce__(self):
        """ This special pickling implementation is needed so huggingface can fingerprint
            datasets that use instances of this class as a generator.
            In addition to the args used to initialize the instance, there is a checksum
            of the source code included to reduce debugging insanity.
        """
        constructor_args = (self.construction_dataset, self.base_tokenizer, self.text_field_name, self.chunk_len, self.min_stride, self.max_waste, self.verbose)
        cls = self.__class__
        # F-You huggingface Dataset caching!
        return (cls._reconstruct, (cls, constructor_args, MODULE_CHECKSUM))

    def __getstate__(self) -> Dict[str, Any]:
        state: Dict[str, Any] = self.__dict__.copy()
        # Remove the non-serializable s3_client from the state
        if '_current_item_generator' in state:
            del state['_current_item_generator']
        # F-You huggingface Dataset caching!
        state['MODULE_CHECKSUM'] = MODULE_CHECKSUM
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the s3_client
        self._current_item_generator= None





