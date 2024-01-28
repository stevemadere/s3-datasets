import sys
import os
import re


sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tokengenerators import (DSItem, TextDS2TokensGenerator)
from typing import cast
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

use_ascii_tokenizer = False if os.path.exists(huggingface_model_path+'/tokenizer.json')  else True

tokenizer: PreTrainedTokenizerFast= \
    cast(PreTrainedTokenizerFast,ASCIITokenizer()) if use_ascii_tokenizer \
    else AutoTokenizer.from_pretrained(huggingface_model_path, use_fast=True)




# yield_tokenized_chunks_from_text_item
def test_yield_tokenized_chunks_from_text_item(single_text_item_dataset) -> None:
    tokenizer.pad_token = tokenizer.eos_token
    the_document = single_text_item_dataset[0]["text"]

    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(single_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    item:DSItem = cast(DSItem, single_text_item_dataset[0])
    chunks:list[DSItem] = [item for item in generator.yield_tokenized_chunks_from_text_item(item)]
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
    tokenizer.pad_token = tokenizer.eos_token
    the_document = single_text_item_dataset[0]["text"]

    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(single_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)

    iterable_ds:datasets.IterableDataset = datasets.IterableDataset.from_generator(generator)

    chunks:list[DSItem] = [cast(DSItem,item) for item in iterable_ds]

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
    tokenizer.pad_token = tokenizer.eos_token
    texts = [dsi["text"] for dsi in  multiple_text_item_dataset]

    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)

    iterable_ds:datasets.IterableDataset = datasets.IterableDataset.from_generator(generator)

    chunks:list[DSItem] = [cast(DSItem,item) for item in iterable_ds]
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



def test_text_ds_2_tokens_generator_exhaustion(multiple_text_item_dataset) -> None:
    chunk_len=4096
    min_stride = 64
    max_waste = 64
    generator:TextDS2TokensGenerator = TextDS2TokensGenerator(multiple_text_item_dataset,tokenizer, chunk_len=chunk_len, min_stride= min_stride, max_waste=max_waste)
    tokens_ds= datasets.Dataset.from_generator(generator)

    num_items:int = 0
    for ds_item in tokens_ds:
        num_items += 1
    expected_num_items = 83 if use_ascii_tokenizer  else 26
    assert num_items  == expected_num_items

