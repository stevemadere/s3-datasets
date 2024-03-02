import torch
from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset, IterableDataset, Features, Sequence, Value
from typing import NewType, Dict, List, Any, Iterator, cast
from functools import total_ordering

""" work status:

- Need to add and test ability to get and set a cursor.
Setting the cursor will necessitate either passing it to the constructor or re-initializing source_dataset

Check to see if construction_dataset responds to set_cursor and get_cursor with this pattern:
if hasattr(obj, 'the_method') and callable(getattr(obj, 'the_method')):
    obj.the_method()

- Need to add and test estimate_available_chunks()

"""


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


# convenience method to convert an indexable dataset to an iterable dataset
def dataset_to_iterable(dataset: Dataset, offset: int = 0) -> Iterator[DSItem]:
    #print(f"class of dataset being converted to iterable is {dataset.__class__}")
    for i in range(offset,len(dataset)):
        yield cast(DSItem,dataset[i])

class RewindBuffer:
    past_items:List
    max_size: int
    index_offset:int

    def __init__(self, max_size) -> None:
        self.past_items = []
        self.max_size = max_size
        self.index_offset = 1

    def add(self, item: Any) -> None:
        self.past_items.append(item)
        if len(self.past_items) > self.max_size:
            self.past_items.pop(0)
            self.index_offset += 1

    def current_range(self):
        return [self.index_offset, self.index_offset+ len(self.past_items)]

    def contains(self,index: int) -> bool:
        return index in range(*self.current_range())

    def get(self,index) -> Any:
        internal_index = index - self.index_offset
        if internal_index < 0 or internal_index >= len(self.past_items):
            raise ValueError("index outside of buffer")
        else:
            return self.past_items[internal_index]


class AddressableWrapOfIterableDataset:
    source_dataset:IterableDataset
    max_rewind:int
    rewind_buffer: RewindBuffer

    def __init__(self, iterable_dataset: IterableDataset, max_rewind = 1024):
        self.source_dataset = iterable_dataset
        self.rewind_buffer = RewindBuffer(max_rewind)

    # brackets operator
    def __getitem__(self, index: int) -> DSItem:
        if self.rewind_buffer.contains(index):
            return self.rewind_buffer.get(index)
        buffer_range = self.rewind_buffer.current_range()
        if buffer_range[0] < index:
            raise ValueError("index preceeds maximum rewind")

        items_to_skip = index - buffer_range[1]
        assert items_to_skip >= 0

        item:DSItem|None = None
        while items_to_skip > 0:
            try:
                item =  cast(DSItem,next(iter(self.source_dataset)))
            except StopIteration:
                raise IndexError
            self.rewind_buffer.add(item)
            items_to_skip -= 1
        assert item != None
        return item



# a json serializable struct with a source cursor and a chunk index
# they should be comparable with a > operator
@total_ordering
class DSGeneratorCursor:
    source_index: int
    chunk_index: int

    def __init__(self, source_index: int= 0, chunk_index: int = 0):
        if source_index < 0:
            raise ValueError("source_index must be >= 0")
        if chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        self.source_index = source_index
        self.chunk_index = chunk_index

    # < cohparison operator
    def __lt__(self, other: 'DSGeneratorCursor') -> bool:
        if self.source_index == other.source_index:
            return self.chunk_index < other.chunk_index
        else:
            return self.source_index < other.source_index

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DSGeneratorCursor):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.source_index == other.source_index and self.chunk_index == other.chunk_index

    def incr_chunk_index(self) -> int:
        self.chunk_index += 1
        return self.chunk_index

    def incr_source_index(self) -> int:
        self.source_index += 1
        self.chunk_index = 0
        return self.source_index

    def to_dict(self) -> dict:
        return {"source_index": self.source_index, "chunk_index": self.chunk_index}

    @classmethod
    def from_dict(cls, d: dict) -> 'DSGeneratorCursor':
        return cls(d["source_index"], d["chunk_index"])


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

        s3_dataset = S3TextDataset(my_bucket_name, prefix=key_prefix)
        ds_dict = s3_dataset.train_test_split(test=0.05)
        ds_generator = TextDS2TokensGenerator(ds_dict["train"], my_tokenizer, text_field_name = "text", chunk_len = 4096, min_stride = 64, max_waste = 64)

        training_dataset = IterableDataset.from_generator(ds_generator, features=ds_generator.features())


        # or for restartable training
        train_docs_dataset = S3TextDataset(my_bucket_name, dataset_id = '/datasets/BigTrainDS.json.gz')
        saved_cursor_dict = json.load(open("saved_cursor.json", "r"))
        saved_cursor = DSGeneratorCursor.from_dict(saved_cursor_dict)
        ds_generator2 = TextDS2TokensGenerator(train_docs_dataset, my_tokenizer, text_field_name = "text", chunk_len = 4096, min_stride = 64, max_waste = 64, cursor = saved_cursor)
        training_dataset2 = IterableDataset.from_generator(ds_generator2, features=ds_generator.features())
        do_some_training_and_checkpoint(training_dataset2)
        cursor = ds_generator2.get_cursor()
        json.dump(cursor.to_dict(), open("saved_cursor.json", "w"))


    """
    construction_dataset: Dataset|IterableDataset
    source_dataset: Dataset|AddressableWrapOfIterableDataset
    base_tokenizer: PreTrainedTokenizerFast
    text_field_name: str
    chunk_len: int
    min_stride: int
    max_waste: int
    include_all_keys: bool
    max_rewind: int
    _cursor: DSGeneratorCursor
    _current_item_chunks: list[DSItem]|None

    def __init__(self,
                 source_dataset: Dataset|IterableDataset,
                 base_tokenizer: PreTrainedTokenizerFast,
                 text_field_name: str = "text",
                 chunk_len: int = 4096,
                 min_stride: int = 64,
                 max_waste: int = 64,
                 verbose: bool = False,
                 include_all_keys: bool = False,
                 max_rewind: int = 1024
                 ) -> None:

        assert min_stride < chunk_len

        self.construction_dataset = source_dataset
        self.max_rewind = max_rewind
        self.base_tokenizer = base_tokenizer
        self.text_field_name = text_field_name
        self.chunk_len = chunk_len
        self.min_stride = min_stride
        self.max_waste = max_waste
        self.verbose = verbose
        self.include_all_keys = include_all_keys
        if isinstance(source_dataset, IterableDataset):
            self.source_dataset = AddressableWrapOfIterableDataset(source_dataset, max_rewind = self.max_rewind)
        else:
            self.source_dataset = source_dataset

        self._cursor = DSGeneratorCursor(0,0)
        self._current_item_chunks = None
        self.set_cursor(self._cursor)
        assert self._current_item_chunks

    def set_cursor(self, cursor: DSGeneratorCursor):
        if not (self._current_item_chunks and self._cursor.source_index == cursor.source_index):
            self._get_source_item_at(cursor.source_index)
        assert self._current_item_chunks and self._cursor.source_index == cursor.source_index
        if cursor.chunk_index >= len(self._current_item_chunks):
            raise IndexError(f"cursor {cursor.to_dict().__repr__()} addresses chunk out of range for item with {len(self._current_item_chunks)} chunks")
        else:
            self._cursor.chunk_index = cursor.chunk_index


    def get_cursor(self) -> DSGeneratorCursor:
        return self._cursor

    def _features_dict(self, force:bool = False) -> dict[str,Any]:
        if self.include_all_keys and not force:
            raise RuntimeError(f"Cannot predict the features that will be returned from {self.__class__} when include_all_keys option is enabled")
        fd: dict[str, Any] = { "slice_index": Value("int64") }
        tokenizer_output_fields = ['input_ids', 'labels', 'attention_mask']
        for field_name in tokenizer_output_fields:
            fd[field_name] = Sequence(feature= Value("int64"), length = self.chunk_len)
        return fd

    def features(self, force: bool = False) -> Features:
        fd = self._features_dict(force)
        return Features(fd)

    def tokenized_chunks_from_text_item(self, text_item: DSItem) -> list[DSItem]:
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
        tokenized_chunks:list[DSItem] = []
        for chunk_slice in slices:
            generated_item:DSItem = DSItem({"input_ids":None})
            for k,v in tokens.items():
                if k == "length":
                    continue # do not include the length field in the yielded items
                values_slice = v[chunk_slice].copy()
                t = torch.tensor(values_slice)
                generated_item[k]=t
                if k == "input_ids":
                    generated_item["labels"] = torch.tensor(values_slice.copy())
            if self.include_all_keys:
                # Also include all members of the original text dataset item except for the text
                for k,v in text_item.items():
                    #print(f"in tokenized_chunks_from_text_item: considering duplication of key '{k}'")
                    if not ( k in generated_item or k == self.text_field_name):
                        #print(f"in tokenized_chunks_from_text_item: duplicating key '{k}' with value '{v}'")
                        generated_item[k] = v
            generated_item["slice_index"] = slice_index
            tokenized_chunks.append(generated_item)
            slice_index+=1
        return tokenized_chunks

    # Need a callable form to make huggingface Dataset.from_generator(me) happy
    def __call__(self):
        iter = self.__iter__()
        for x  in iter:
            yield(x)

    def __iter__(self):
        return self

    def _get_source_item_at(self, index: int):
        #print(f"source_dataset: {self.source_dataset.__repr__()}")
        if index < 0:
            raise IndexError
        item = cast(DSItem, self.source_dataset[index] )

        assert isinstance(item[self.text_field_name], str)

        item_chunks = self.tokenized_chunks_from_text_item(item)
        self._cursor = DSGeneratorCursor(index,0)
        self._current_item_chunks = item_chunks
        #print(f"current_source_item keys: {current_source_item.keys()}")

    def __next__(self) -> DSItem:
        assert self._current_item_chunks
        assert len(self._current_item_chunks) > 0
        while True:
            while self._cursor.chunk_index >= len(self._current_item_chunks):
                new_source_index = self._cursor.source_index+1
                try:
                    self._get_source_item_at(new_source_index)
                    assert self._cursor.source_index == new_source_index
                    assert self._cursor.chunk_index == 0
                except IndexError:
                    # print(f"setting cursor to {new_source_index},0 at end of iteration")
                    self._cursor = DSGeneratorCursor(new_source_index,0)
                    raise StopIteration
            item = self._current_item_chunks[self._cursor.chunk_index]
            self._cursor.incr_chunk_index()
            return item

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
    @classmethod
    def _reconstruct(cls, args, checksum):
        if checksum: # silence the unused params warning
            pass
        obj = cls(*args)
        return obj

    def __reduce__(self):
        """ This special pickling implementation is needed so huggingface can fingerprint
            datasets that use instances of this class as a generator.
            In addition to the args used to initialize the instance, there is a checksum
            of the source code included to reduce debugging insanity.
        """
        constructor_args = (self.construction_dataset, self.base_tokenizer, self.text_field_name, self.chunk_len, self.min_stride, self.max_waste, self.verbose, self.max_rewind)
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


