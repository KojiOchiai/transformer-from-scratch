import os
import warnings
from pathlib import Path
from typing import Generator

import torch
import torch.nn as nn
import torchmetrics
import torchtext.datasets as datasets  # type: ignore
from datasets import Dataset as HFDataset  # type: ignore # From Huggingface
from datasets import load_dataset  # type: ignore # From Huggingface
from tokenizers import Tokenizer  # type: ignore # From Huggingface
from tokenizers.models import WordLevel  # type: ignore # From Huggingface
from tokenizers.pre_tokenizers import Whitespace  # type: ignore # From Huggingface
from tokenizers.trainers import WordLevelTrainer  # type: ignore # From Huggingface
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import TranslationDataset, causal_mask
from model import build_transformer


def get_all_sentences(ds: HFDataset, lang: str) -> Generator[str, None, None]:
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config: dict, ds: HFDataset, lang: str) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        # # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
