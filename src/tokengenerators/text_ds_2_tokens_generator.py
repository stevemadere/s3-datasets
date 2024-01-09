import torch
from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset, IterableDataset
from typing import NewType, Dict, List, Any, Iterator, cast

DSItem = NewType('DSItem',Dict[str,Any])

#TokenizerListResult = NewType('TokenizerListResult', Dict[str,List])
#TokenizerTensorResult = NewType('TokenizerTensorResult', dict[str,torch.Tensor])

def dataset_to_iterable(dataset: Dataset) -> Iterator[DSItem]:
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
    _current_item_generator:Iterator[DSItem]|None

    def __init__(self,source_dataset: Dataset|IterableDataset, base_tokenizer: PreTrainedTokenizerFast, text_field_name: str = "text", chunk_len: int = 4096, min_stride: int = 64, max_waste: int = 64) -> None:
        assert min_stride < chunk_len
        self.construction_dataset = source_dataset
        self.source_dataset = cast(Iterator[DSItem],self.construction_dataset) if isinstance(self.construction_dataset, IterableDataset) else dataset_to_iterable(self.construction_dataset)
        self.base_tokenizer = base_tokenizer
        self.text_field_name = text_field_name
        self.chunk_len = chunk_len
        self.min_stride = min_stride
        self.max_waste = max_waste
        self._current_item_generator = None


    def yield_tokenized_chunks_from_text_item(self, text_item: DSItem) -> Iterator[DSItem]:
        text: str = text_item[self.text_field_name]
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
            yield generated_item

    # Need a callable form to make huggingface Dataset.from_generator(me) happy
    def __call__(self):
        iter = self.__iter__()
        for x  in iter:
            yield(x)

    def __iter__(self):
        self._current_item_generator=None
        return self

    def _get_next_source_item(self):
        source_item:Any = next(self.source_dataset)
        current_source_item = cast(DSItem,source_item)
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

        # print(f"remaining_partitions = {remaining_partitions}")

        # After the first chunk is laid down, determine what is left to cover by remaining chunks and allocate it
        ideal_partition_len: float = remaining_tokens / remaining_partitions
        # print(f"ideal_partition_len = {ideal_partition_len}")
        optimal_stride: float = chunk_len - ideal_partition_len
        # print(f"optimal_stride = {optimal_stride}")
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

    # custom pickling methods to enable fingerprinting for Dataset with_transform compatibility 
    def __reduce__(self):
        return (self.__class__, (self.construction_dataset, self.base_tokenizer, self.text_field_name, self.chunk_len, self.min_stride, self.max_waste))

    def __getstate__(self) -> Dict[str, Any]:
        state: Dict[str, Any] = self.__dict__.copy()
        # Remove the non-serializable s3_client from the state
        if '_current_item_generator' in state:
            del state['_current_item_generator']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the s3_client
        self._current_item_generator= None





