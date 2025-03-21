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


def get_ds(config: dict) -> tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    # it only has the train split, so we divide it overselves
    ds_raw = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val__ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val__ds_size])

    train_ds = TranslationDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq"],
    )
    val_ds = TranslationDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq"],
    )

    # Find the maximum length of eath sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> nn.Module:
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq"],
        config["seq"],
        d_model=config["d_model"],
    )
    return model
