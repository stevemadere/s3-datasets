import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from datasets import Dataset, IterableDataset, Features, Sequence, Value
from typing import NewType, Dict, List, Any, Iterator, cast
from functools import total_ordering
import copy
import typedjson


# Why do I need a checksum of my module source code? you may ask.
#  Well, let me tell you a story of hair-pulling insanity during debugging when
#  huggingface Dataset decides to cache the implementation of my class
#  *in the freaking dataset cache*  and uses old versions of the code (some, but not all of it!)
#  while executing my dataaset generator even as I'm debugging and modifying my source code.
#  To prevent this insanity-inducing, infuriating behavior, I need to make sure the
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
    """ Computes a checksum of the named file.
        Used for detecting source code changes.
    """
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Compute checksum using __file__ to get the current module's path
module_path = os.path.abspath(__file__)

# to notify huggingface Dataset caching when this code changes
MODULE_CHECKSUM = compute_checksum(module_path)


DSItem = NewType('DSItem',Dict[str,Any])

#TokenizerListResult = NewType('TokenizerListResult', Dict[str,List])
#TokenizerTensorResult = NewType('TokenizerTensorResult', dict[str,torch.Tensor])


class RewindBuffer:
    """ Holds a limited cache of dataset items preceding the cursor to allow for a some
        rewind capacity in otherwise unindexable IterableDataset instances.
    """
    past_items: list
    max_size: int
    index_offset:int

    def __init__(self, max_size: int) -> None:
        assert max_size > 0
        self.past_items = []
        self.max_size = max_size
        self.index_offset = 0

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
    """ Wraps an IterableDataset to give it addressable Dataset semantics.
        Uses a rewind buffer to allow addressing at offsets with a limited
        range prior to the current iteration.  Uses forward scanning to
        address after the current iteration.
    """
    source_iterator: Iterator
    max_rewind:int
    rewind_buffer: RewindBuffer

    def __init__(self, iterable_dataset: IterableDataset, max_rewind = 1024):
        self.source_iterator = iter(iterable_dataset)
        self.rewind_buffer = RewindBuffer(max_rewind)

    # brackets operator
    def __getitem__(self, index: int) -> DSItem:
        if self.rewind_buffer.contains(index):
            return self.rewind_buffer.get(index)
        buffer_range = self.rewind_buffer.current_range()
        if index < buffer_range[0]:
            raise ValueError("index preceeds maximum rewind")

        items_to_skip = (index - buffer_range[1])+1
        assert items_to_skip > 0

        item:DSItem|None = None
        while items_to_skip > 0:
            try:
                item =  cast(DSItem,next(self.source_iterator))
            except StopIteration:
                raise IndexError
            self.rewind_buffer.add(item)
            items_to_skip -= 1
        assert not item == None
        return item



@total_ordering
class DSGeneratorCursor:
    """
       A cursor into a tokenized and chunked dataset generation stream.

       Tracks the document in a source dataset of ordered documents
       and a specific chunk of tokens produced from that document.

       Can be saved by serializing the result of cursor_dict = cursor.to_dict() and
       can be restored by deserializing that cursor_dict and calling
       dsg_cursor = DSGeneratorCursor.from_dict(cursor_dict)
    """

    source_index: int
    chunk_index: int
    exhausted: bool # indicates that iteration hit the end since cursor was set

    def __init__(self, source_index: int= 0, chunk_index: int = 0):
        if source_index < 0:
            raise ValueError("source_index must be >= 0")
        if chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        self.source_index = source_index
        self.chunk_index = chunk_index
        self.exhausted = False

    # < comparison operator
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

    def incr(self, chunk_limit:int) -> 'DSGeneratorCursor':
        new_chunk_index = self.chunk_index + 1
        if new_chunk_index < chunk_limit:
            self.chunk_index = new_chunk_index
        else:
            self.incr_source_index()
        return self

    def incr_source_index(self) -> int:
        self.source_index += 1
        self.chunk_index = 0
        return self.source_index

    def to_dict(self) -> dict[str,int]:
        return {"source_index": self.source_index, "chunk_index": self.chunk_index}

    def save_to_file_path(self, file_path:str) -> None:
        json_string = typedjson.class_instance_to_json(self)
        with open(file_path, "w") as f:
            f.write(json_string)

    @classmethod
    def from_dict(cls, d: dict[str,int]) -> 'DSGeneratorCursor':
        return cls(source_index=d["source_index"], chunk_index = d["chunk_index"])

    @classmethod
    def from_file_path(cls, file_path:str) -> 'DSGeneratorCursor':
        """
        deserialized_thing = json.load(open(file_path, "r"))
        # let's get pydantic, shall we?
        if (isinstance(deserialized_thing, dict) and
                all(isinstance(key,str) and isinstance(value,int) for key,value in deserialized_thing.items())):
            cursor_dict = cast(dict[str,int], deserialized_thing)
            return cls.from_dict(cursor_dict)
        else:
            raise ValueError("object in cursor file must be of type dict[str,int]")
        """
        with open(file_path, "r") as f:
            instance = typedjson.class_instance_from_json(cls, f.read())
        return instance


class TextDS2TokensGenerator:
    """
        Produces generators that can be used with IterableDataset.from_generator()
        to tokenize a series of text Dataset/IterableDataset items.
        Instantiate a member of this class with an underlying dataset (which can be an
        IterableDataset or a Dataset) containing text documents to be tokenized.
        This generator generates token sequences of a specified fixed length lazily,
        thus saving significant local storage and network bandwidth when your dataset
        of text is absolutely huge.  It works particularly well in combination with
        S3TextDataset which lazily loads text documents from an S3 bucket.

        example usage:

        s3_dataset = S3TextDataset(my_bucket_name, prefix=key_prefix)
        ds_dict = s3_dataset.train_test_split(test=0.05)
        ds_generator = TextDS2TokensGenerator(ds_dict["train"], my_tokenizer, text_field_name = "text", chunk_len = 4096, min_stride = 64, max_waste = 64)

        training_dataset = IterableDataset.from_generator(ds_generator, features=ds_generator.features())


        # or for restartable training
        train_docs_dataset = S3TextDataset(my_bucket_name, dataset_id = '/datasets/BigTrainDS.json.gz')
        cursor_file_name="train_dataset_cursor.json"
        old_cursor_file_path = os.path.join(previous_checkpoint_dir, cursor_file_name)
        saved_cursor_dict = json.load(open(old_cursor_file_path, "r"))
        saved_cursor = DSGeneratorCursor.from_dict(saved_cursor_dict)
        ds_generator2 = TextDS2TokensGenerator(train_docs_dataset, my_tokenizer, text_field_name = "text", chunk_len = 4096, min_stride = 64, max_waste = 64, cursor = saved_cursor)
        training_dataset2 = IterableDataset.from_generator(ds_generator2, features=ds_generator.features())
        new_checkpoint_dir = do_some_training_and_checkpoint(training_dataset2)
        new_cursor_file_path = os.path.join(new_checkpoint_dir, cursor_file_name)
        json.dump(ds_generator2.get_cursor().to_dict(), open(new_cursor_file_path, "w"))

    """
    construction_dataset: Dataset|IterableDataset
    source_dataset: Dataset|AddressableWrapOfIterableDataset
    base_tokenizer: PreTrainedTokenizerBase
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
                 base_tokenizer: PreTrainedTokenizerBase,
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

        self._current_item_chunks = None
        self.set_cursor(DSGeneratorCursor(0,0))

    def get_cursor(self) -> DSGeneratorCursor:
        """
        Gets the generator's cursor that can be used later to return it to the
        generation state where it currently is.

        This is can be useful when resuming a training process after a crash or
        intentional pause.  e.g.  When your spot-market cloud GPU gets yanked and
        you subsequently want to resume from your most recent checkpoint.

        usage:
        # before a crash
        training_dataset = IterableDataset.from_generator(my_generator)
        do_some_training_and_make_a_checkpoint(my_model, training_dataset)
        cursor = my_generator.get_cursor()
        save_cursor_to_checkpoint(cursor)

        # resuming training after a crash
        pre_crash_cursor = get_cursor_from_checpoint()
        my_generator.set_cursor(pre_crash_cursor)
        training_dataset = IterableDataset.from_generator(my_generator)
        do_some_training_from_checkpoint_and_checkpoint_again(training_dataset)
        cursor = my_generator.get_cursor()
        save_cursor_to_checkpoint(cursor)
        # ad infinitum
        """
        cursor_copy = copy.deepcopy(self._cursor) # ensure the caller can't mess up my internal state with mods to the cursor
        return cursor_copy

    def set_cursor(self, new_cursor: DSGeneratorCursor):
        """
        Sets the generator's cursor to allow for resumption of generation from a previously recorded state.

        The cursor should only be a value that was previously returned by a call to
        get_cursor() on an equivalent generator.

        Confirms that the cursor is valid by retrieving the source data item, tokenizing it,
        and splitting into chunks of tokens.  If the cursor is invalid, an IndexError will
        be raised.

        see get_cursor() documentation for rationale and example usage.

        """
        if self._current_item_chunks and self._cursor and self._cursor.source_index == new_cursor.source_index:
            if self.verbose:
                print(f"using existing item_chunks at {new_cursor.source_index}")
            item_chunks = self._current_item_chunks
        else:
            if self.verbose:
                print(f"loading item_chunks at {new_cursor.source_index}")
            item_chunks = self._get_item_chunks_at(new_cursor.source_index) # this may raise IndexError depending on source_index
        if new_cursor.chunk_index >= len(item_chunks):
            raise IndexError(f"cursor {new_cursor.to_dict().__repr__()} addresses chunk out of range for item with {len(item_chunks)} chunks")
        else:
            self._current_item_chunks = item_chunks
            self._cursor = copy.deepcopy(new_cursor)
            self.exhausted = False

    def features(self, force: bool = False) -> Features:
        """ Generate a Features object suitable for passing to IterableDataset.from_generator().
            If the generator was constructed with the "include_all_keys" option set, it will be
            inaccurate so the default behavior is to raise an error in that case.
            Using the force option to this method skips raising an error and just returns
            a possibly incomplete description of the generated features.

            example usage :

            tokens_generator = TextDS2TokensGenerator(text_dataset, tokenizer)
            tokenized_ds = IterableDataset.from_generator(tokens_generator, features = tokens_generator.features())

        """
        fd = self._features_dict(force)
        return Features(fd)

    def estimate_available_chunks(self,
                                  max_relative_uncertainty: float|None =None,
                                  max_uncertainty:int|None = None) -> tuple[int|None,float]:
        import numpy as np

        if max_relative_uncertainty and max_uncertainty:
            raise ValueError("at most, one of max_relative_uncertainty and max_uncertainty can be specified")

        if isinstance(self.construction_dataset,Dataset):
            finite_dataset: Dataset = cast(Dataset, self.construction_dataset)
        else:
            raise TypeError("This generator is sourced from an IterableDataset.  Since its size cannot be known, no estimate can be made of the chunks available from it")

        source_dataset_size = len(finite_dataset)
        relative_uncertainty:float = 1.0
        uncertainty = max_uncertainty+1.0 if max_uncertainty else 2.0e9
        min_samples = 10
        sample_sum = 0
        samples = []
        estimated_mean:None|float = None
        for sample_position in range(source_dataset_size):
            if sample_position > min_samples and (
                 (max_relative_uncertainty and relative_uncertainty < max_relative_uncertainty) or
                 (max_uncertainty and uncertainty < max_uncertainty)):
                break
            chunks = self._get_item_chunks_at(sample_position)
            sample = len(chunks)
            if self.verbose:
                print(f"sample #{sample_position} : {sample}")
            samples.append(sample)
            sample_sum += sample
            sample_size:float = float(len(samples))
            uncertainty:float = sample_sum
            if sample_position > 5:
                sample_std = np.std(samples,ddof=1)
                uncertainty =  float(sample_std) * float(np.sqrt(sample_size))
            estimated_mean = float(sample_sum) / sample_size
            relative_uncertainty = uncertainty / sample_sum
            if self.verbose:
                print(f"estimated_mean: {estimated_mean}, relative_uncertainty {relative_uncertainty}")

        estimated_chunks:int|None = None
        if len(samples) >= source_dataset_size:
            relative_uncertainty = 0.0
            estimated_chunks = sample_sum
        else:
            estimated_chunks = int(estimated_mean*source_dataset_size) if estimated_mean else None
        return (estimated_chunks, relative_uncertainty)


    def _advance_cursor_without_read(self, loaded_chunks_len:int) -> None:
        assert self._current_item_chunks and self._cursor
        next_cursor:DSGeneratorCursor = copy.deepcopy(self._cursor)
        next_cursor.incr(chunk_limit = loaded_chunks_len)
        if not next_cursor.source_index == self._cursor.source_index:
            self._current_item_chunks = None
        self._cursor = next_cursor


    def _features_dict(self, force:bool = False) -> dict[str,Any]:
        if self.include_all_keys and not force:
            raise RuntimeError(f"Cannot predict the features that will be returned from {self.__class__} when include_all_keys option is enabled")
        fd: dict[str, Any] = { "slice_index": Value("int64") }
        tokenizer_output_fields = ['input_ids', 'labels', 'attention_mask']
        for field_name in tokenizer_output_fields:
            fd[field_name] = Sequence(feature= Value("int64"), length = self.chunk_len)
        return fd

    def _tokenized_chunks_from_text_item(self, text_item: DSItem) -> list[DSItem]:
        text: str = text_item[self.text_field_name]
        tokens: BatchEncoding = self.base_tokenizer(text, max_length = 1024*1024*1024, return_length = True, truncation=True)
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
                    if not ( k in generated_item or k == self.text_field_name):
                        generated_item[k] = v
            generated_item["slice_index"] = slice_index
            tokenized_chunks.append(generated_item)
            slice_index+=1
        return tokenized_chunks

    # Need a callable form to make huggingface IterableDataset.from_generator(me) happy
    def __call__(self):
        iter = self.__iter__()
        for x  in iter:
            yield(x)

    def __iter__(self):
        return self

    def _get_item_chunks_at(self, index: int):
        if index < 0:
            raise IndexError
        item = cast(DSItem, self.source_dataset[index] )
        assert isinstance(item[self.text_field_name], str)
        item_chunks = self._tokenized_chunks_from_text_item(item)
        return item_chunks

    def __next__(self) -> DSItem:
        if not self._cursor:
            self._cursor = DSGeneratorCursor(0,0)
            self._current_item_chunks = None # force a reload of self._current_item_chunks below
        while True:
            try:
                self.set_cursor(self._cursor) # ensure _current_item_chunks is populated and range checks are made
                assert self._current_item_chunks
                item = self._current_item_chunks[self._cursor.chunk_index]
                self._advance_cursor_without_read(len(self._current_item_chunks))
            except IndexError:
                self.exhausted = True
                raise StopIteration
            assert self._cursor
            return item

    @staticmethod
    def  calculate_optimal_step_size(num_tokens: int, chunk_len: int, min_stride: int, max_waste: int) -> float:
        assert num_tokens > chunk_len + max_waste

        # there is always at least one chunk, take that off and see what's left to cover

        remaining_tokens = num_tokens - chunk_len
        remaining_partitions: int = (((remaining_tokens-max_waste) -1)  // (chunk_len - min_stride) ) + 1


        # After the first chunk is laid down, determine what is left to cover by remaining chunks and allocate it
        ideal_partition_len: float = remaining_tokens / remaining_partitions
        optimal_stride: float = chunk_len - ideal_partition_len
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
        # the MODULE_CHECKSUM is there to notify huggingface Dataset caching when the code changes
        return (cls._reconstruct, (cls, constructor_args, MODULE_CHECKSUM))

    def __getstate__(self) -> Dict[str, Any]:
        state: Dict[str, Any] = self.__dict__.copy()
        # notify huggingface Dataset caching when the code changes
        state['MODULE_CHECKSUM'] = MODULE_CHECKSUM
        # remove cursor information from signature
        del state['_cursor']
        del state['_current_item_chunks']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not '_cursor' in self.__dict__:
            self.__dict__['_cursor'] = DSGeneratorCursor(0,0)
        if not '_current_item_chunks' in self.__dict__:
            self.__dict__['_current_item_chunks'] = None


