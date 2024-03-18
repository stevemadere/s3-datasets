import sys
import os
import re
import copy


sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tokengenerators import DSItem, TextDS2TokensGenerator, AddressableWrapOfIterableDataset, DSGeneratorCursor
from typing import cast, Any
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import datasets
import pytest

# calculate_optimal_step_size
def test_calculate_optimal_step_size_overflow() -> None:
    # Confirm that it expects num_tokens to exceed chunk_len + max_waste
    with pytest.raises(AssertionError):
       TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 100, chunk_len=90, min_stride=20, max_waste=11)

def test_calculate_optimal_step_size_two_mostly_overlapping_chunks() -> None:
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 100, chunk_len=90, min_stride=20, max_waste=9) <= 10.0
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 100, chunk_len=90, min_stride=20, max_waste=9) > 9.0

def test_calculate_optimal_step_size_three_chunks_no_waste() -> None:
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 100, chunk_len=50, min_stride=10, max_waste=9) > 24.0
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 100, chunk_len=50, min_stride=10, max_waste=9) < 26.0


def test_calculate_optimal_step_size_minimal_chunks_with_waste() -> None:
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 100, chunk_len=50, min_stride=10, max_waste=12) > 39.0
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 100, chunk_len=50, min_stride=10, max_waste=12) <= 40.0

def test_calculate_optimal_step_size_many_chunks_no_waste() -> None:
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 1000, chunk_len=50, min_stride=10, max_waste=9) > 39.0
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 1000, chunk_len=50, min_stride=10, max_waste=9) <= 40.0

def test_calculate_optimal_step_size_many_chunks_with_waste() -> None:
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 1008, chunk_len=50, min_stride=10, max_waste=9) > 39.0
    assert TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 1008, chunk_len=50, min_stride=10, max_waste=9) <= 40.0

def test_calculate_optimal_step_size_many_chunks_with_waste2() -> None:
    oss = TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = 1022, chunk_len=50, min_stride=10, max_waste=9)
    assert oss < 39.0
    assert oss > 38.0

def test_calculate_optimal_step_realistic_scenario() -> None:
    num_tokens, chunk_len, min_stride, max_waste = 12121, 4096, 64, 64
    oss = TextDS2TokensGenerator.calculate_optimal_step_size(num_tokens = num_tokens, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    assert oss > 4012
    assert oss <  4013

# calculate_slices
def test_calculate_slices_two_mostly_overlapping_chunks() -> None:
    assert TextDS2TokensGenerator.calculate_slices(num_tokens = 100, chunk_len= 90, optimal_step_size=9.5) == [slice(0, 90), slice(10, 100)]

def test_calculate_slices_three_chunks_no_waste() -> None:
    assert TextDS2TokensGenerator.calculate_slices(num_tokens = 100, chunk_len= 50, optimal_step_size=25.0) == [slice(0, 50), slice(25, 75), slice(50,100)]

def test_calculate_slices_minimal_chunks_with_waste() -> None:
    assert TextDS2TokensGenerator.calculate_slices(num_tokens = 100, chunk_len= 50, optimal_step_size=40.0) == [slice(0, 50), slice(40, 90)]

def test_calculate_slices_realistic_scenario() -> None:
    assert TextDS2TokensGenerator.calculate_slices(num_tokens = 12121, chunk_len= 4096, optimal_step_size=4012.5) == [slice(0, 4096), slice(4013, 8109), slice(8025, 12121)]



# If the huge huggingface models are not available locally, use this sort of mock of a tokenizer
class ASCIITokenizer:
    eos_token:int
    pad_token:int|None
    padding_side:str|None

    def __init__(self):
        self.eos_token = 1
        self.pad_token = None
        self.padding_side = 'left'

    @staticmethod
    def encode(text: str, max_length:int|None = None) -> list[int]:
        return [ord(c) for c in text][0:max_length]

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return "".join([chr(c) for c in tokens])

    def __call__(self, text: str, max_length: int|None = None , return_length: bool = False, padding:None|str|bool = False, truncation:None|bool = None) -> dict[str,list]:
        if truncation: # silence the unused params warning
            pass
        input_ids:list[int] = ASCIITokenizer.encode(text, max_length)
        num_tokens: int = len(input_ids)
        if max_length and num_tokens < max_length and padding == 'max_length':
            if self.pad_token == None:
                raise ValueError("Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`")
            pad_size = max_length - num_tokens
            pad = [self.pad_token for _ in range(pad_size)]
            if self.padding_side == 'left':
                input_ids = [*pad, *input_ids]
            else:
                input_ids = [*input_ids, *pad]
            num_tokens = len(input_ids) # this is not redundant because input_ids got modified above
        attention_mask:list[int] = [1 for _ in range(num_tokens)]
        length:list[int] = [num_tokens]
        if return_length:
            return { "input_ids": input_ids, "attention_mask":attention_mask, "length":length }
        else:
            return { "input_ids": input_ids, "attention_mask":attention_mask }




my_dirname:str = os.path.dirname(__file__)
huggingface_model_path=f"{my_dirname}/../../vastai/huggingface/Mistral-7B-v0.1"

ascii_tokenizer:PreTrainedTokenizerFast = cast(PreTrainedTokenizerFast,ASCIITokenizer())

use_ascii_tokenizer = True # False if os.path.exists(huggingface_model_path+'/tokenizer.json')  else True

tokenizer: PreTrainedTokenizerFast= \
    ascii_tokenizer if use_ascii_tokenizer \
    else AutoTokenizer.from_pretrained(huggingface_model_path, use_fast=True)

tokenizer.pad_token = tokenizer.eos_token



# tokenized_chunks_from_text_item
def test_tokenized_chunks_from_text_item(single_text_item_dataset) -> None:
    the_document = single_text_item_dataset[0]["text"]

    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(single_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    item:DSItem = cast(DSItem, single_text_item_dataset[0])
    chunks:list[DSItem] = [item for item in generator._tokenized_chunks_from_text_item(item)]
    assert len(chunks[0]["input_ids"]) == chunk_len
    for chunk in chunks:
        assert len(chunk["input_ids"]) == chunk_len
        assert len(chunk["attention_mask"]) == chunk_len

    beginning_text:str = tokenizer.decode(chunks[0]["input_ids"])
    # remove a leading start_token sequence ("<s> ") if there is one
    beginning_text = re.sub("^<s> ", "", beginning_text)
    ending_text:str = tokenizer.decode(chunks[-1]["input_ids"])
    assert beginning_text == the_document[0:len(beginning_text)]
    assert ending_text == the_document[-len(ending_text):]
    if len(chunks)>2:
        second_text = tokenizer.decode(chunks[1]["input_ids"])
        position_of_second_text_in_long_text = the_document.find(second_text)
        assert position_of_second_text_in_long_text > max_waste
        position_of_ending_text = the_document.find(ending_text)
        assert position_of_second_text_in_long_text < position_of_ending_text


# TextDS2TokensGenerator
def test_text_ds_2_tokens_generator_one_doc(single_text_item_dataset) -> None:
    the_document = single_text_item_dataset[0]["text"]

    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(single_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)

    iterable_ds:datasets.IterableDataset = datasets.IterableDataset.from_generator(generator)

    assert not generator.exhausted
    chunks:list[DSItem] = [cast(DSItem,item) for item in iterable_ds]
    assert generator.exhausted

    assert len(chunks[0]["input_ids"]) == chunk_len
    for chunk in chunks:
        assert len(chunk["input_ids"]) == chunk_len
        assert len(chunk["attention_mask"]) == chunk_len

    beginning_text:str = tokenizer.decode(chunks[0]["input_ids"])
    # remove a leading start_token sequence ("<s> ") if there is one
    beginning_text = re.sub("^<s> ", "", beginning_text)
    ending_text:str = tokenizer.decode(chunks[-1]["input_ids"])
    assert beginning_text == the_document[0:len(beginning_text)]
    assert ending_text == the_document[-len(ending_text):]
    if len(chunks)>2:
        second_text = tokenizer.decode(chunks[1]["input_ids"])
        position_of_second_text_in_long_text = the_document.find(second_text)
        assert position_of_second_text_in_long_text > max_waste
        position_of_ending_text = the_document.find(ending_text)
        assert position_of_second_text_in_long_text < position_of_ending_text


# TextDS2TokensGenerator
def test_text_ds_2_tokens_generator_multi_doc(multiple_text_item_dataset) -> None:
    texts = [dsi["text"] for dsi in  multiple_text_item_dataset]

    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)

    iterable_ds:datasets.IterableDataset = datasets.IterableDataset.from_generator(generator)

    assert not generator.exhausted
    chunks:list[DSItem] = [cast(DSItem,item) for item in iterable_ds]
    assert generator.exhausted

    assert len(chunks[0]["input_ids"]) == chunk_len
    for chunk in chunks:
        assert len(chunk["input_ids"]) == chunk_len
        assert len(chunk["attention_mask"]) == chunk_len

    beginning_text:str = tokenizer.decode(chunks[0]["input_ids"])
    # remove a leading start_token sequence ("<s> ") if there is one
    beginning_text = re.sub("^<s> ", "", beginning_text)
    assert beginning_text == texts[0][0:len(beginning_text)]
    ending_text:str = tokenizer.decode(chunks[-1]["input_ids"])
    position_of_ending_text = texts[-1].find(ending_text)
    assert position_of_ending_text > 0
    assert position_of_ending_text <= len(texts[-1])-len(ending_text)
    if len(chunks)>2:
        second_text = tokenizer.decode(chunks[1]["input_ids"])
        position_of_second_text_in_long_text = texts[0].find(second_text)
        assert position_of_second_text_in_long_text > max_waste
        assert position_of_second_text_in_long_text < len(texts[0])-chunk_len


def test_features_dict_standard_keys(multiple_text_item_dataset) -> None:
    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)

    features_dict: dict[str,Any] = generator._features_dict()
    tokenizer_output_fields=["input_ids", "labels", "attention_mask"]
    extra_fields = ["slice_index"]
    assert set(features_dict.keys()) == set(tokenizer_output_fields + extra_fields)
    for feature in tokenizer_output_fields:
        assert features_dict[feature].length == chunk_len

def test_features_dict_include_all_keys_raises(multiple_text_item_dataset) -> None:
    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste,include_all_keys=True)

    # confirm that calling _features_dict raises a RuntimeError with a message matching include_all_keys
    with pytest.raises(RuntimeError) as exc:
        generator._features_dict()
    assert "include_all_keys" in str(exc.value)

    # confirm that calling _features_dict with force does not raise that error
    features_dict = generator._features_dict(force=True)
    assert isinstance(features_dict, dict)


def test_text_ds_2_tokens_generator_exhaustion(multiple_text_item_dataset) -> None:
    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    tokens_ds= datasets.IterableDataset.from_generator(generator)

    num_items:int = 0
    for ds_item in tokens_ds:
        assert not generator.exhausted
        assert ds_item
        num_items += 1

    assert generator.exhausted
    expected_num_items = 83 if use_ascii_tokenizer  else 26
    assert num_items  == expected_num_items

def test_text_ds_2_tokens_get_cursor(multiple_text_item_dataset) -> None:
    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    tokens_ds= datasets.IterableDataset.from_generator(generator)

    cursor = None
    prev_cursor = None
    num_items:int = 0
    for ds_item in tokens_ds:
        assert not generator.exhausted
        assert ds_item
        num_items += 1
        cursor = generator.get_cursor()
        # print(f"num_items is {num_items}, cursor is {cursor.to_dict().__repr__()} prev_cursor is {prev_cursor.__repr__()}")
        if prev_cursor:
            assert cursor > prev_cursor
        prev_cursor =  copy.deepcopy(cursor)

    assert generator.exhausted

    cursor = generator.get_cursor()
    assert cursor and cursor.source_index == len(multiple_text_item_dataset)
    expected_num_items = 83 if use_ascii_tokenizer  else 26
    assert num_items  == expected_num_items

def test_addressable_wrap_of_iterable_dataset(multiple_text_item_dataset) -> None:
    text_items = []
    for i in range(len(multiple_text_item_dataset)):
        text_items.append(multiple_text_item_dataset[i]['text'])

    iterable_ds:datasets.IterableDataset = multiple_text_item_dataset.to_iterable_dataset()

    addressable_wrap = AddressableWrapOfIterableDataset(iterable_ds)
    for n in range(len(multiple_text_item_dataset)):
        item = addressable_wrap[n]
        assert item
        assert item['text']
        assert len(item['text']) == len(text_items[n])
        assert item['text'] == text_items[n]


def test_text_ds_2_tokens_get_cursor_with_iterable_ds(multiple_text_item_dataset) -> None:
    text_items = []
    for i in range(len(multiple_text_item_dataset)):
        text_items.append(multiple_text_item_dataset[i]['text'])

    iterable_ds:datasets.IterableDataset = multiple_text_item_dataset.to_iterable_dataset()

    chunk_len=4096
    min_stride = 64
    max_waste = 64

    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(iterable_ds, tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    tokens_ds= datasets.IterableDataset.from_generator(generator)

    cursor = None
    prev_cursor = None
    num_items:int = 0
    for ds_item in tokens_ds:
        assert not generator.exhausted
        assert ds_item
        num_items += 1
        cursor = generator.get_cursor()
        # print(f"num_items is {num_items}, cursor is {cursor.to_dict().__repr__()} prev_cursor is {prev_cursor.__repr__()}")
        if prev_cursor:
            assert cursor > prev_cursor
        prev_cursor =  copy.deepcopy(cursor)

    assert generator.exhausted

    cursor = generator.get_cursor()
    assert cursor and cursor.source_index == len(multiple_text_item_dataset)
    expected_num_items = 83 if use_ascii_tokenizer  else 26
    assert num_items  == expected_num_items

def test_set_cursor(multiple_text_item_dataset) -> None:
    import torch

    chunk_len=4096
    min_stride = 64
    max_waste = 64

    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset, tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)


    # print(f"original cursor {generator.get_cursor().to_dict().__repr__()}")
    tokens_ds= datasets.IterableDataset.from_generator(generator)
    #print(f"after creating dataset {generator.get_cursor().to_dict().__repr__()}")

    items_at_cursors: list[tuple] = []
    cursor:DSGeneratorCursor = generator.get_cursor()
    prev_cursor:DSGeneratorCursor|None = None
    num_items:int = 0
    for ds_item in tokens_ds:
        assert not generator.exhausted
        # print(f"item at cursor {cursor.to_dict().__repr__()}")
        assert ds_item
        num_items += 1
        items_at_cursors.append((cursor, ds_item))
        # print(f"num_items is {num_items}, cursor is {cursor.to_dict().__repr__()} prev_cursor is {prev_cursor.__repr__()}")
        cursor = generator.get_cursor()
        if prev_cursor:
            assert cursor > prev_cursor
        prev_cursor =  copy.deepcopy(cursor)

    assert generator.exhausted

    generator2:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset, tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    for (cursor, ds_item) in reversed(items_at_cursors):
        # print(f"about to set cursor to {cursor.to_dict().__repr__()}")
        generator2.set_cursor(cursor)
        item_at_cursor = next(iter(generator2))
        # print(item_at_cursor.keys());
        assert torch.allclose(item_at_cursor["input_ids"], ds_item["input_ids"])

def test_set_cursor_with_iterable_source(multiple_text_item_dataset) -> None:
    import torch

    chunk_len=4096
    min_stride = 64
    max_waste = 64

    iterable_ds:datasets.IterableDataset = multiple_text_item_dataset.to_iterable_dataset()

    chunk_len=4096
    min_stride = 64
    max_waste = 64

    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(iterable_ds, tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    tokens_ds= datasets.IterableDataset.from_generator(generator)

    assert not generator.exhausted


    # print(f"original cursor {generator.get_cursor().to_dict().__repr__()}")
    tokens_ds= datasets.IterableDataset.from_generator(generator)
    #print(f"after creating dataset {generator.get_cursor().to_dict().__repr__()}")

    items_at_cursors: list[tuple] = []
    cursor:DSGeneratorCursor = generator.get_cursor()
    prev_cursor:DSGeneratorCursor|None = None
    num_items:int = 0

    for ds_item in tokens_ds:
        assert not generator.exhausted
        # print(f"item at cursor {cursor.to_dict().__repr__()}")
        assert ds_item
        num_items += 1
        items_at_cursors.append((cursor, ds_item))
        # print(f"num_items is {num_items}, cursor is {cursor.to_dict().__repr__()} prev_cursor is {prev_cursor.__repr__()}")
        cursor = generator.get_cursor()
        if prev_cursor:
            assert cursor > prev_cursor
        prev_cursor =  copy.deepcopy(cursor)

    assert generator.exhausted

    iterable_ds2:datasets.IterableDataset = multiple_text_item_dataset.to_iterable_dataset()
    generator2:TextDS2TokensGenerator = TextDS2TokensGenerator(iterable_ds2, tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    for (cursor, ds_item) in reversed(items_at_cursors):
        # print(f"about to set cursor to {cursor.to_dict().__repr__()}")
        generator2.set_cursor(cursor)
        item_at_cursor = next(iter(generator2))
        # print(item_at_cursor.keys());
        assert torch.allclose(item_at_cursor["input_ids"], ds_item["input_ids"])


def test_estimate_available_chunks_smoke(multiple_text_item_dataset) -> None:
    chunk_len=4096
    min_stride = 64
    max_waste = 64

    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset, tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)

    max_relative_uncertainty = 0.1
    allowable_slop = 0.5
    (chunks_estimate, uncertainty) = generator.estimate_available_chunks(max_relative_uncertainty = max_relative_uncertainty)

    assert chunks_estimate
    assert uncertainty
    # print(f"uncertainty returned was {uncertainty}")

    actual_number_of_chunks = len(list(generator))

    ridiculously_low_estimate = 0
    ridiculously_high_estimate = actual_number_of_chunks * 2
    assert chunks_estimate > ridiculously_low_estimate
    assert chunks_estimate < ridiculously_high_estimate

    excessively_high_estimate = actual_number_of_chunks * (1.0+max_relative_uncertainty*(1.0+allowable_slop))
    excessively_low_estimate = actual_number_of_chunks * (1.0-max_relative_uncertainty*(1.0+allowable_slop))
    assert chunks_estimate > excessively_low_estimate
    assert chunks_estimate < excessively_high_estimate

if os.environ['S3_TEXT_DATASET_BUCKET']:

    def test_estimate_available_chunks_real_world() -> None:
        from s3datasets import S3TextDataset

        chunk_len=4096
        min_stride = 64
        max_waste = 64
        bucket_name = os.environ['S3_TEXT_DATASET_BUCKET']
        bucket_prefix = os.environ.get('S3_TEXT_DATASET_PREFIX','')
        reasonable_chunks_per_text = float(os.environ.get('EXPECTED_CHUNKS_PER_TEXT','16'))
        max_relative_uncertainty = float(os.environ.get('AVAILABLE_CHUNKS_UNCERTAINTY','0.05'))

        large_text_ds = S3TextDataset.from_bucket(bucket_name, prefix=bucket_prefix)
        num_texts = len(large_text_ds)

        generator:TextDS2TokensGenerator = TextDS2TokensGenerator(large_text_ds, ascii_tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste, verbose=False)

        allowable_slop = 0.5
        (chunks_estimate, uncertainty) = generator.estimate_available_chunks(max_relative_uncertainty = max_relative_uncertainty)

        assert chunks_estimate
        assert uncertainty
        # print(f"chunks_estimate and uncertainty returned was ({chunks_estimate}, {uncertainty})")


        reasonable_num_chunks = reasonable_chunks_per_text * num_texts

        ridiculously_low_estimate = 0
        ridiculously_high_estimate = reasonable_num_chunks * 2
        assert chunks_estimate > ridiculously_low_estimate
        assert chunks_estimate < ridiculously_high_estimate

        excessively_high_estimate = reasonable_num_chunks * (1.0+max_relative_uncertainty*(1.0+allowable_slop))
        excessively_low_estimate = reasonable_num_chunks * (1.0-max_relative_uncertainty*(1.0+allowable_slop))
        assert chunks_estimate > excessively_low_estimate
        assert chunks_estimate < excessively_high_estimate


def test_ds_generator_cursor_save_and_load() -> None:
    import random
    import tempfile
    import os
    original_cursor = DSGeneratorCursor(source_index=random.randint(1,1000), chunk_index= random.randint(0,7))
    # make this a safe temp file name that will be auto-deleted when the process exits
    with tempfile.NamedTemporaryFile(mode='w',delete=True) as tf:
        cursor_file_path =  tf.name
        original_cursor.save_to_file_path(cursor_file_path)
        assert os.path.exists(cursor_file_path)
        restored_cursor = DSGeneratorCursor.from_file_path(cursor_file_path)
    assert not os.path.exists(cursor_file_path)
    assert restored_cursor == original_cursor

