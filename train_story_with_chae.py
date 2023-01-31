#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on text translation.
"""
# You can also adapt this script on your own text translation task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--predict_with_generate",
        type=bool,
        default=do_write_file,
        help="",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default='data/{}/train.csv'.format(data_path),
        help="A csv or a json file containing the training data."
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=2,  # None
        help="Number of beams to use for evaluation. This argument will be "
             "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
             "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
             "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
             "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
             "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
             "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
             "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help="Whether to pad all samples to model maximum sentence "
             "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
             "efficient on GPU but very bad for TPU.",
    )
    parser.add_argument(
        "--validation_file", type=str, default='data/{}/valid.csv'.format(data_path),
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default='data/{}/emo_data/test.csv'.format(data_name),
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--pre_file", type=str, default='data/{}/pre.csv'.format(data_path),
        help="A csv or a json file containing the prediction data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=model_name_or_path,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=n_epoch,  # 3
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='checkpoint/{}'.format(data_name),
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=33, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--do_predict", type=str, default=do_predict,
                        help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks

    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")

    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def eval_story(dataloader, prediction_file=None, label_file=None, num=None):
    model.eval()
    if prediction_file != None:
        output_prediction_file = os.path.join(args.output_dir, prediction_file)
    if label_file != None:
        output_label_file = os.path.join(args.output_dir, label_file)

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        # "num_beams": args.num_beams,
        "num_beams": 1,  # num_beams=1: greedy search / top-k
        "do_sample": True,  # beam_search = False / top-k = True
        "top_k": 50,  # 50 / 30
        "temperature": 1.2,  # 0.6 / 0.8 / 1 / 1.2
    }

    story_pre_list = []
    story_labels_list = []
    story_chae_list = []
    new_inputs = ''
    new_labels = ''
    new_chae = []

    for step, batch in enumerate(dataloader):
        if step % 4 == 0:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        with torch.no_grad():
            generated_tokens, emo_output = accelerator.unwrap_model(model).generate(
                # transformers/generation_utils.py 1789
                torch.cat([input_ids, batch["chae"]], 1) if batch["chae"] is not None else input_ids,
                attention_mask=torch.cat([attention_mask, batch["chae_mask"]], 1) if batch[
                                                                                         "chae"] is not None else attention_mask,
                output_attentions=True,
                chae=batch["chae"], chae_mask=batch["chae_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens,
                                                   skip_special_tokens=True)  # batch_decode: convert ids to str
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            eval_bar.update(1)

            # predictions
            predictions = [pred.strip() for pred in decoded_preds]
            with open(output_prediction_file, "a", encoding="utf-8") as writer:
                writer.write("\n".join(predictions) + '\n')

            # controllability
            decoded_chae = tokenizer.batch_decode(batch["chae"])
            decoded_input_ids = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            decoded_chae, decoded_input_ids = postprocess_text(decoded_chae, decoded_input_ids)
            de_input_ids = [''.join(input_ids).strip() for input_ids in decoded_input_ids]
            de_labels = [''.join(label).strip() for label in decoded_labels]  # <-------write labels
            with open(output_prediction_file.split('.')[0] + '_controllability.txt', "a",
                      encoding="utf-8") as writer:
                for i in range(len(decoded_chae)):
                    writer.write('chae: ' + decoded_chae[i] + '\n')
                    writer.write('input: ' + input_ids[i] + '\n')
                    writer.write('label: ' + labels[i] + '\n')
                    writer.write('prediction: ' + predictions[i] + '\n')
                    if emo_output is not None:
                        writer.write('pre emo1: ' + EMO_idx[emo_output[0].tolist()[i]] + '\n')
                        writer.write('pre emo2: ' + EMO_idx[emo_output[1].tolist()[i]] + '\n\n')
            if step % 4 == 0:
                # step = 4, 8, 12, ...
                if step != 0:
                    story_pre_list.append(new_inputs)  # [[sent1,...,sent5], [sent1,...,sent5]]
                    story_labels_list.append(new_labels)  # [[sent1,...,sent5], [sent1,...,sent5]]
                    story_chae_list.append(new_chae)  # [[chae2,3,4,5], [chae2,3,4,5]]

                if num == 0:
                    with open(output_label_file, "a", encoding="utf-8") as writer:
                        writer.write(new_labels + '\n')

                with open(output_prediction_file, "a", encoding="utf-8") as writer:
                    writer.write(new_inputs + '\n')

                # step = 0, 4, 8, ...
                new_inputs = de_input_ids[0]
                new_labels = de_input_ids[0]
                new_chae = []  #

            with open(output_prediction_file.split('.')[0] + '_controllability.txt', "a",
                      encoding="utf-8") as writer:
                if step % 4 == 0:
                    writer.write('input: ' + new_inputs + '\n')
                writer.write('chae: ' + decoded_chae[0] + '\n')
                writer.write('prediction: ' + predictions[0] + '\n\n')
                writer.write('label: ' + de_labels[0] + '\n')
                if emo_output is not None:
                    writer.write('pre emo1: ' + EMO_idx[emo_output[0].tolist()[0]] + '\n')
                    writer.write('pre emo2: ' + EMO_idx[emo_output[1].tolist()[0]] + '\n\n')

            new_inputs += ' ' + predictions[0]
            tokenized_inputs = tokenizer(new_inputs, max_length=args.max_target_length, padding=False, truncation=True,
                                         return_tensors='pt')
            input_ids = tokenized_inputs['input_ids'].to(batch["input_ids"].device)
            attention_mask = tokenized_inputs['attention_mask'].to(batch["input_ids"].device)
            new_labels += ' ' + de_labels[0]  # <-------write labels
            new_chae.append(decoded_chae[0])  # <-------write controllability

            if do_debug:
                if step == 3:
                    break

    return None


def train(completed_steps, epoch):
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs, emo_output = model(**batch)
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)
        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

        if step % 100 == 0:
            if emo_output is None:
                logger.info("epoch {} loss {}: ".format(epoch, loss.item()))
            else:
                logger.info(
                    "epoch {} lm_loss {} emo_loss {} emo_acc {}".format(epoch, loss.item() - emo_output[2].item(),
                                                                        emo_output[2].item(), emo_output[-1]))

        if completed_steps >= args.max_train_steps:
            break

        if do_debug:
            if step == 5:
                break
    return completed_steps


def evaluate(dataloader, prediction_file=None, label_file=None, num=None):
    model.eval()
    if prediction_file != None:
        output_prediction_file = os.path.join(args.output_dir, prediction_file)
    if label_file != None:
        output_label_file = os.path.join(args.output_dir, label_file)

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
    }
    total_eval_loss = 0
    total_eval_emo_loss = 0
    total_eval_emo_acc = 0
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            valid_output, emo_output = model(**batch)
            if emo_output is not None:
                total_eval_emo_loss += emo_output[2].item()
                total_eval_emo_acc += emo_output[-1]
                valid_loss = valid_output.loss - emo_output[2]
            else:
                valid_loss = valid_output.loss
            total_eval_loss += valid_loss.item()
            eval_bar.update(1)

            if args.predict_with_generate:
                generated_outputs, emo_output = accelerator.unwrap_model(model).generate(
                    # transformers/generation_utils.py 1789
                    torch.cat([batch["input_ids"], batch["chae"]], 1) if batch["chae"] is not None else batch[
                        "input_ids"],
                    attention_mask=torch.cat([batch["attention_mask"], batch["chae_mask"]], 1) if batch[
                                                                                                      "chae"] is not None else
                    batch["attention_mask"],
                    output_attentions=True, return_dict_in_generate=True,
                    chae=batch["chae"], chae_mask=batch["chae_mask"],
                    **gen_kwargs,
                )
                generated_tokens = generated_outputs.sequences
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                decoded_preds = tokenizer.batch_decode(generated_tokens,
                                                       skip_special_tokens=True)  # batch_decode: convert ids to str
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                predictions = [pred.strip() for pred in decoded_preds]

                decoded_chae = tokenizer.batch_decode(batch["chae"])
                decoded_input_ids = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                decoded_chae, decoded_input_ids = postprocess_text(decoded_chae, decoded_input_ids)
                input_ids = [''.join(input_ids).strip() for input_ids in decoded_input_ids]
                labels = [''.join(label).strip() for label in decoded_labels]

                if args.predict_with_generate:
                    # write labels
                    if num == 0:
                        labels = [''.join(label).strip() for label in decoded_labels]
                        with open(output_label_file, "a", encoding="utf-8") as writer:
                            writer.write("\n".join(labels) + '\n')

                    # write predictions
                    with open(output_prediction_file, "a", encoding="utf-8") as writer:
                        writer.write("\n".join(predictions) + '\n')

                    # write controllability
                    with open(output_prediction_file.split('.')[0] + '_controllability.txt', "a",
                              encoding="utf-8") as writer:
                        for i in range(len(decoded_chae)):
                            writer.write('chae: ' + decoded_chae[i] + '\n')
                            writer.write('input: ' + input_ids[i] + '\n')
                            writer.write('label: ' + labels[i] + '\n')
                            writer.write('prediction: ' + predictions[i] + '\n')
                            if emo_output is not None:
                                writer.write('pre emo1: ' + EMO_idx[emo_output[0].tolist()[i]] + '\n')
                                writer.write('pre emo2: ' + EMO_idx[emo_output[1].tolist()[i]] + '\n\n')

            if do_debug:
                if step == 15:
                    break

    # '''
    logger.info('showing evaluation results of the last batch ')
    decoded_chae = tokenizer.batch_decode(batch["chae"])
    decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    decoded_chae, decoded_inputs = postprocess_text(decoded_chae, decoded_inputs)
    inputs = [''.join(input).strip() for input in decoded_inputs]
    labels = [''.join(label).strip() for label in decoded_labels]
    predictions = [pred.strip() for pred in decoded_preds]
    for i in range(len(labels)):
        logger.info('eval example {}'.format(i))
        logger.info('input: {}'.format(inputs[i]))
        logger.info('chae: {}'.format(decoded_chae[i]))
        logger.info('label: {}'.format(labels[i]))
        logger.info('prediction: {}'.format(predictions[i]))
        if emo_output is not None:
            logger.info('pre emo1: {}'.format(EMO_idx[emo_output[0].tolist()[i]]))
            logger.info('pre emo2: {}'.format(EMO_idx[emo_output[1].tolist()[i]]))
    # '''

    valid_average_loss = total_eval_loss / len(dataloader)
    valid_average_emo_loss = total_eval_emo_loss / len(dataloader)
    valid_average_emo_acc = total_eval_emo_acc / len(dataloader)
    ppl = np.exp(valid_average_loss)

    if emo_output is None:
        logger.info({"valid_average_lm_loss": valid_average_loss, "ppl": ppl})
    else:
        logger.info(
            {"valid_average_loss": valid_average_loss, 'valid_average_emo_loss': valid_average_emo_loss, "ppl": ppl,
             "emo_acc": valid_average_emo_acc})

    return valid_average_loss


def main():
    global model, tokenizer, postprocess_text, metric, config, args, model, train_dataloader, \
        accelerator, optimizer, lr_scheduler, progress_bar, eval_bar

    # Parse the arguments
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state)
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    # Created a handler for writing to log file
    file_handler = logging.FileHandler(filename='logs/{}/log_{}.txt'.format(data_name, name))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        if args.pre_file is not None:
            data_files["prediction"] = args.pre_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    num = 0
    for model_name_or_path in model_name_or_path_list:
        if args.config_name:
            config = AutoConfig.from_pretrained(model_name_or_path)
        elif model_name_or_path:
            config = AutoConfig.from_pretrained(model_name_or_path)
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        config.do_copy = do_copy
        config.do_emo = do_emo
        config.emo_weight = do_emo_weight
        config.do_similarity = do_similarity

        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        elif model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=not args.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        if model_name_or_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForSeq2SeqLM.from_config(config)

        # add special tokens
        # special_tokens_dict = {'additional_special_tokens': ["<soc>", "<eoc>", "<soa>", "<eoa>", "<soe>", "<eoe>", "<NoAction>"]}
        special_tokens_dict = {'additional_special_tokens': ["<SEP>", "<soc>", "<sep>", "<soa>", "<soe>",
                                                             "<no_action>"]}
        tokenizer.add_special_tokens(special_tokens_dict)

        model.resize_token_embeddings(len(tokenizer))

        # Set decoder_start_token_id
        if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            assert (
                    args.target_lang is not None and args.source_lang is not None
            ), "mBart requires --target_lang and --source_lang"
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.target_lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        prefix = args.source_prefix if args.source_prefix is not None else ""

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names

        # For translation we set the codes of our source and target languages (only useful for mBART, the others will
        # ignore those attributes).
        if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if args.source_lang is not None:
                tokenizer.src_lang = args.source_lang
            if args.target_lang is not None:
                tokenizer.tgt_lang = args.target_lang

        # Get the language codes for input/target.
        # source_lang = args.source_lang.split("_")[0]
        # target_lang = args.target_lang.split("_")[0]

        padding = "max_length" if args.pad_to_max_length else False

        # Temporarily set max_target_length for training.
        max_target_length = args.max_target_length
        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_function_chae(examples):
            # I love you <pad> <pad> <SEP> <soc> char1 <soa> act1 <soe> emo1 <SEP> <soc> char2 <soa> act2 <soe> emo2
            inputs = examples['inputs']
            characters = examples['characters']
            emotions = examples['emotions']
            actions = examples['actions']
            targets = examples['outputs']
            new_inputs = []
            for i in inputs:
                new_inputs.append(tokenizer.bos_token + ' ' + i)
            model_inputs = tokenizer(new_inputs, max_length=args.max_source_length, padding=padding, truncation=True,
                                     add_special_tokens=False)
            ChAE = []
            emo_label = []
            # raise KeyError
            for i in range(len(characters)):
                chae = ''
                e_label = []
                for char in eval(characters[i]):
                    a = ''
                    for sent_a in eval(actions[i]):
                        if sent_a[char] == []:
                            a += '<no_action>' + ' '
                        else:
                            a += ' <sep> '.join(sent_a[char]) + ' '
                    e = ''
                    for sent_e in eval(emotions[i]):
                        if sent_e[char] == []:
                            e += 'neutral' + ' '
                            e_label.append(EMO_MAP['neutral'])
                        else:
                            e += sent_e[char][0] + ' '
                            e_label.append(EMO_MAP[sent_e[char][0]])
                    chae += '<SEP>' + ' <soc> ' + char + ' <soa> ' + a + '<soe> ' + e
                chae += tokenizer.eos_token

                if len(eval(characters[i])) < 2:
                    e_label.append(EMO_MAP['neutral'])
                ChAE.append(chae.strip())
                emo_label.append(e_label)

            if do_emo:
                model_inputs['emo_labels'] = emo_label
            model_ChAE = tokenizer(ChAE, max_length=args.max_source_length, padding=padding, truncation=True,
                                   add_special_tokens=False)
            model_inputs['chae'] = model_ChAE["input_ids"]
            model_inputs['chae_mask'] = model_ChAE["attention_mask"]
            labels = tokenizer(targets, max_length=args.max_target_length, padding=padding, truncation=True,
                               add_special_tokens=False)
            if padding == "max_length" or args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        with accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function_chae,  # preprocess_function / preprocess_function_all / preprocess_function_outline
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if accelerator.use_fp16 else None,
            )

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        metric = load_metric("packages/sacrebleu.py")

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]

            return preds, labels

        if do_self_predict:
            logger.info("*** Self Predict ***")
            pre_dataset = processed_datasets["prediction"]
            pre_dataloader = DataLoader(pre_dataset, collate_fn=data_collator, batch_size=1)
            pre_dataloader = accelerator.prepare(pre_dataloader)
            logger.info('Run checkpoint : {}'.format(model_name_or_path))
            predict_epoch = model_name_or_path.split('_')[-1][5:]
            if do_base_or_finetune == 'finetune':
                prediction_file = '{}_epoch{}_self_pre_beam2.txt'.format(name, predict_epoch)
                label_file = '{}_labels.txt'.format(name)
            else:
                prediction_file = '{}_base_predictions.txt'.format(model_name_or_path.split('/')[-1])
                label_file = '{}_base_labels.txt'.format(model_name_or_path.split('/')[-1])
            eval_bar = tqdm(range(len(pre_dataloader)), disable=not accelerator.is_local_main_process,
                            desc='Eval (epoch #{})'.format(predict_epoch))
            evaluate(pre_dataloader, prediction_file, label_file, num)
            num += 1
            if num == len(model_name_or_path_list):
                return

        if args.do_predict:
            logger.info("*** Predict ***")
            test_dataset = processed_datasets["test"]
            batch_size = args.per_device_eval_batch_size
            test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size)
            test_dataloader = accelerator.prepare(test_dataloader)
            logger.info('Run checkpoint : {}'.format(model_name_or_path))
            predict_epoch = model_name_or_path.split('_')[-1][5:]
            if do_base_or_finetune == 'finetune':
                prediction_file = '{}_epoch{}_predictions.txt'.format(name, predict_epoch)
                label_file = '{}_labels.txt'.format(name)
            else:
                prediction_file = '{}_base_predictions.txt'.format(model_name_or_path.split('/')[-1])
                label_file = '{}_base_labels.txt'.format(model_name_or_path.split('/')[-1])
            eval_bar = tqdm(range(len(test_dataloader)), disable=not accelerator.is_local_main_process,
                            desc='Eval (epoch #{})'.format(predict_epoch))
            evaluate(test_dataloader, prediction_file, label_file, num)
            num += 1
            if num == len(model_name_or_path_list):
                return

        if do_eval_story:
            logger.info("*** Eval Story ***")
            test_dataset = processed_datasets["test"]
            test_dataloader = DataLoader(test_dataset, collate_fn=data_collator,
                                         batch_size=1)
            test_dataloader = accelerator.prepare(test_dataloader)
            logger.info('Run checkpoint : {}'.format(model_name_or_path))
            predict_epoch = model_name_or_path.split('_')[-1][5:]
            if do_base_or_finetune == 'finetune':
                prediction_file = '{}_epoch{}_story_topk50_temp12.txt'.format(name, predict_epoch)
                # prediction_file = '{}_epoch{}_story_greedy.txt'.format(name, predict_epoch)
                label_file = '{}_story_labels.txt'.format(name)
            else:
                prediction_file = '{}_base_story.txt'.format(model_name_or_path.split('/')[-1])
                label_file = '{}_base_story_labels.txt'.format(model_name_or_path.split('/')[-1])
            eval_bar = tqdm(range(len(test_dataloader)), disable=not accelerator.is_local_main_process,
                            desc='Eval (epoch #{})'.format(predict_epoch))
            eval_story(test_dataloader, prediction_file, label_file, num)
            num += 1
            if num == len(model_name_or_path_list):
                return

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.

    completed_steps = 0
    best_valid_loss = float('inf')
    best_epoch = -1
    for epoch in range(args.num_train_epochs):
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process,
                            desc='Train (epoch #{})'.format(epoch))
        eval_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process,
                        desc='Eval (epoch #{})'.format(epoch))

        completed_steps = train(completed_steps, epoch)
        valid_loss = evaluate(eval_dataloader)

        if args.output_dir is not None and do_debug != True:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir + '/' + name + '_epoch' + str(epoch),
                                            save_function=accelerator.save)
            logger.info("save parameters of epoch {}".format(epoch))
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir + '/' + name + '_epoch' + str(epoch))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
            logger.info("The best epoch is : {}".format(best_epoch))


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

    # You should update this to your particular problem to have better documentation of `model_type`
    MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
    MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    time = '04061730'
    data_name = 'n_to_1_new'
    n_epoch = 3
    do_debug = False
    do_copy = True
    do_emo = True
    do_emo_weight = True
    do_similarity = False

    do_predict = False
    do_eval_story = True
    do_write_file = False
    do_self_predict = False
    do_base_or_finetune = 'finetune'
    data_path = data_name

    name = '{}_{}'.format(time, data_name)
    # model_name_or_path = 'facebook/bart-large-cnn'
    if do_emo:
        data_path += '/emo_data'
        model_name_or_path = 'checkpoint/n_to_1_new/04051820_n_to_1_new_epoch0'  # 只copy chae，且p_gen修改

    model_name_or_path_list = [model_name_or_path]

    EMO_MAP = {
        "joy": 0,
        "trust": 1,
        "fear": 2,
        "surprise": 3,
        "sadness": 4,
        "disgust": 5,
        "anger": 6,
        "anticipation": 7,
        "neutral": 8
    }

    EMO_idx = {v: k for k, v in EMO_MAP.items()}

    if do_predict or do_self_predict or do_eval_story:
        # fine-tune
        if do_base_or_finetune == 'finetune':
            if do_predict:
                model_name_or_path_list = ['checkpoint/{}/{}_epoch{}'.format(data_name, name, i) for i in
                                           range(n_epoch)]  # 设置从哪个epoch文件开始
            else:
                model_name_or_path_list = ['checkpoint/{}/{}_epoch{}'.format(data_name, name, 0)]

        prediction_file = 'generated_predictions.txt'
        label_file = 'generated_labels.txt'
    main()
