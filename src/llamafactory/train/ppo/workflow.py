# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/ppo.py
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

from typing import TYPE_CHECKING, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ...data import RMinference, get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_ref_model, create_reward_model
from .trainer import CustomPPOTrainer,RminferenceTrainer
from tqdm.auto import tqdm
from accelerate import Accelerator
import torch

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_ppo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    #tokenizer_module = load_tokenizer(model_args)
    #tokenizer = tokenizer_module["tokenizer"]
    #template = get_template_and_fix_tokenizer(tokenizer, data_args)
    #dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    #model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    #tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    #data_collator = MultiModalDataCollatorForSeq2Seq(template=template, model=model, **tokenizer_module)

    # Create reference model and reward model
    #ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    #tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    model = None
    tokenizer_module,tokenizer,reward_model = create_reward_model(model, model_args, finetuning_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    #dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    data_collator = RMinference(template=template, model=model, **tokenizer_module)


    def get_dataset_module_with_idx(*args, **kwargs):
        dm = get_dataset(*args, **kwargs)  # 你的原接口
        # dm 里通常有 'train_dataset' 和 'eval_dataset'
        dm["train_dataset"] = add_idx(dm["train_dataset"])
        return dm
    def add_idx(ds):
    # ds: Dataset 或 IterableDataset
    # with_indices=True 会把当前样本在这个 split 里的位置当 idx 传进来
        return ds.map(lambda example, idx: {"idx": idx}, with_indices=True)
    dataset_module = get_dataset_module_with_idx(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    
    import os
    import debugpy

    # Initialize our Trainer

    trainer = RminferenceTrainer(
        model=reward_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    

    accelerator = trainer.accelerator   
    train_dataset = dataset_module["train_dataset"]
    trainer.create_optimizer_and_scheduler(1)
    model     = trainer.model          # 已包装好（DDP / FSDP / DeepSpeed 等）
    optimizer = trainer.optimizer      # create_optimizer_and_scheduler 里创建好的
    scheduler = trainer.lr_scheduler   # 如果你需要学习率调度器

    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    #model = reward_model

    #model, dataloader = accelerator.prepare(model, dataloader)
    model, dataloader = accelerator.prepare(
        model,
        dataloader,
    )
    import json
    from tqdm import tqdm

    model.eval()
    records = []                      # 用于最终保存，每元素 = {"text": str, "reward": float}
    total_batches = len(dataloader)
    all_reward=[]
    all_input_ids = []
    decoded_inputs_list=[]
    for step, batch in enumerate(tqdm(dataloader, disable=not accelerator.is_main_process), start=1):
        with torch.no_grad():
            inference_inputs = {
                "input_ids":      batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "pixel_values":   batch["pixel_values"],
                "image_grid_thw": batch["image_grid_thw"],
            }
            idx = batch["idx"]
            mask = (idx == 5)
            if mask.any():                                    
                selected_ids = inference_inputs["input_ids"][mask]   # shape = (N, L)
                decoded_inputs = tokenizer.batch_decode(
                    selected_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                print(decoded_inputs)

            outputs = model(**inference_inputs,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=False)

            hidden = outputs[2]         # == outputs[2]
            last_idx = inference_inputs["attention_mask"].sum(-1, keepdim=True) - 1
            rewards  = hidden.gather(-1, last_idx).squeeze(-1)        
            rewards, idx = accelerator.gather_for_metrics((rewards, idx))
            all_reward.extend(rewards.cpu().tolist())
            all_input_ids.extend(idx.cpu().tolist())

    assert len(all_input_ids) == len(all_reward), "数量不一致，无法一一对应保存！"

    idx_reward_data = [
        {"idx": int(idx), "reward": float(r)}
        for idx, r in zip(all_input_ids, all_reward)
    ]

    with open(f"{training_args.output_dir}/idx_reward.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(obj, ensure_ascii=False)
                        for obj in idx_reward_data))

    print(f"[rank0] total saved {len(idx_reward_data)} examples to idx_reward.jsonl")

