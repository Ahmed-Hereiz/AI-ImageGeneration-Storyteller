import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)


class ModelArguments:
    def __init__(self, model_name_or_path=None, model_type=None, cache_dir=None):
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.cache_dir = cache_dir


class DataTrainingArguments:
    def __init__(self, train_data_file=None, eval_data_file=None, line_by_line=False, mlm=False, block_size=-1, overwrite_cache=False):
        self.train_data_file = train_data_file
        self.eval_data_file = eval_data_file
        self.line_by_line = line_by_line
        self.mlm = mlm
        self.block_size = block_size
        self.overwrite_cache = overwrite_cache

        
def get_dataset(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache)

    
def train(num_epochs):

    model_args = ModelArguments(model_name_or_path="gpt2", model_type="gpt2")
    data_args = DataTrainingArguments(
        train_data_file="/content/drive/MyDrive/data_nlp/6_genre_clean_training_data.txt",
        eval_data_file="/content/drive/MyDrive/data_nlp/6_genre_eval_data.txt",
        line_by_line=True,
        block_size=512,
        overwrite_cache=True,
    )
    training_args = TrainingArguments(
        output_dir="story_generator_checkpoint",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        logging_steps=500,
        per_device_train_batch_size=4,
        num_train_epochs=num_epochs,
        save_total_limit=1,
        save_steps=1000,
    )

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file.")

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path, config=config)

    special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        "additional_special_tokens": ["<superhero>", "<action>", "<drama>", "<thriller>", "<horror>", "<sci_fi>"],
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=data_args.mlm)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    try:
        if training_args.do_train:
            model_path = model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            trainer.train(model_path=model_path)
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)
    except KeyboardInterrupt:
        print("Saving model that was in the middle of training")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        return

    
    
# train(num_epochs=6)   ## train the model with 6 epochs