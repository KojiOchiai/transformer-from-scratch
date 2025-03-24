import os
import warnings
from pathlib import Path
from typing import Callable, Generator

import torch
import torch.nn as nn
import torchmetrics
from datasets import Dataset as HFDataset  # type: ignore # From Huggingface
from datasets import load_dataset  # type: ignore # From Huggingface
from tokenizers import Tokenizer  # type: ignore # From Huggingface
from tokenizers.models import WordLevel  # type: ignore # From Huggingface
from tokenizers.pre_tokenizers import Whitespace  # type: ignore # From Huggingface
from tokenizers.trainers import WordLevelTrainer  # type: ignore # From Huggingface
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, get_weights_file_path, latest_weights_file_path
from dataset import TranslationDataset, causal_mask
from model import Transformer, build_transformer


def greedy_decode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokennzer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    sos_idx = tokennzer_tgt.token_to_id("[SOS]")
    eos_idx = tokennzer_tgt.token_to_id("[EOS]")

    # Precomute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initiaqlize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) >= max_len:
            break

        # build mask for target
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model: Transformer,
    validation_ds: DataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    print_msg: Callable,
    global_step: int,
    writer: SummaryWriter,
    num_examples: int = 2,
) -> None:
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console winindow width
        with os.popen("stty size", "r") as console:
            _, console_width_str = console.read().split()
            console_width = int(console_width_str)
    except Exception:
        # if we can't get the console width, we use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg("-" * console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

        if writer:
            # Evaluete the character error rate
            # Compute the char error rate
            metric_cer = torchmetrics.CharErrorRate()
            cer = metric_cer(predicted, expected)
            writer.add_scalar("validation cer", cer, global_step)
            writer.flush()

            # Compute the word error rate
            metric_wer = torchmetrics.WordErrorRate()
            wer = metric_wer(predicted, expected)
            writer.add_scalar("validation wer", wer, global_step)
            writer.flush()

            # Compute the BLEU metric
            metric_bleu = torchmetrics.BLEUScore()
            bleu = metric_bleu(predicted, expected)
            writer.add_scalar("validation BLEU", bleu, global_step)
            writer.flush()


def get_all_sentences(ds: HFDataset, lang: str) -> Generator[str, None, None]:
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config: Config, ds: HFDataset, lang: str) -> Tokenizer:
    tokenizer_path = Path(config.tokenizer_file.format(lang))
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


def get_ds(config: Config) -> tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    # it only has the train split, so we divide it overselves
    ds_raw = load_dataset(
        f"{config.datasource}",
        f"{config.lang_src}-{config.lang_tgt}",
        split="train",
    )

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config.lang_tgt)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val__ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val__ds_size])

    train_ds = TranslationDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.lang_src,
        config.lang_tgt,
        config.seq,
    )
    val_ds = TranslationDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.lang_src,
        config.lang_tgt,
        config.seq,
    )

    # Find the maximum length of eath sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config.lang_tgt]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: Config, vocab_src_len: int, vocab_tgt_len: int) -> Transformer:
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config.seq,
        config.seq,
        d_model=config.d_model,
    )
    return model


def train_model(config: Config) -> None:
    # Define the device
    device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device_type}")
    device = torch.device(device_type)
    if device_type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        total_memory = torch.cuda.get_device_properties(device.index).total_memory
        print(f"Device memory: {total_memory / 1024**3} GB")
    elif device_type == "mps":
        print("Device name: <mps>")
    else:
        print("Note: If you have GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: "
            "https://www.youtube.com/watch?v=GMSjDTU8Zlc"
        )
        print(
            "      On a Mac machine, "
            "run: pip3 install --pre torch torchvision torchaudio torchtext "
            "--index-url https://download.pytorch.org/whl/nightly/cpu"
        )

    # Make sure the weight folder exists
    Path(f"{config.datasource}_{config.model_folder}").mkdir(
        parents=True, exist_ok=True
    )

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)
    # Tensorboard
    writer = SummaryWriter(config.experiment_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config.preload
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload)
        if preload
        else None
    )
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting flom scratch")

    loss_fun = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1
    )

    for epoch in range(initial_epoch, config.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (b, seq)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, seq)
            decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, seq, seq)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq, d_model)
            decoder_outout = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq, d_model)
            proj_output = model.project(decoder_outout)  # (B, seq, vocab_size)

            # Compare the outoput with the label
            label: torch.Tensor = batch["label"].to(device)  # (B, seq)

            # Compute the loss using a simple cross entropy
            loss: torch.Tensor = loss_fun(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagete the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of each epoch
        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config.seq,
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
        )

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = Config()
    train_model(config)
