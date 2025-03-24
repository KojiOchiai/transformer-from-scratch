from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    batch_size: int = 8
    num_epochs: int = 30
    lr: float = 1e-4
    seq: int = 350
    d_model: int = 512
    datasource: str = "opus_books"
    lang_src: str = "en"
    lang_tgt: str = "it"
    model_folder: str = "weights"
    model_basename: str = "tmodel_"
    preload: str = "latest"
    tokenizer_file: str = "tokenizer_{0}.json"
    experiment_name: str = "runs/tmodel"


def get_weights_file_path(config: Config, epoch: str) -> str:
    model_folder = f"{config.datasource}_{config.model_folder}"
    model_filename = f"{config.model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config: Config) -> str | None:
    model_folder = f"{config.datasource}_{config.model_folder}"
    model_filename = f"{config.model_basename}*"
    weight_files = list(Path(model_folder).glob(model_filename))
    if len(weight_files) == 0:
        return None
    weight_files.sort()
    return str(weight_files[-1])
