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
