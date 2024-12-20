from huggingface_hub import HfFolder, whoami
hf_token = "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"
HfFolder.save_token(hf_token)

user = whoami()
print(user['name'])

import os
import torch
from datasets import load_dataset
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
BitsAndBytesConfig,
HfArgumentParser,
pipeline,
logging)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig


DatasetName = "N0v4123/ultrachat-10k-chatml-llama-3.2"
MyModelName = "Llama-3.2-1B-chat-FineTuned"
ModelName = "meta-llama/Llama-3.2-1B"

# Lora dimensions
LoraR = 2

# Alpha for scaling
LoraAlpha = 16

# Dropout rate
LoraDropout = 0.1

# Quantization using Bits and bytes
Use4bit = True

# From
Bnb4bitQuantDType = "float16"

# To
Bnb4bitQuantType = "nf4"

# Nested Quant
UseNestedQuant = False

# Training Arguments
OutputDir = "./results"

# Number of training epochs
NumTrainEpochs = 2

# Enable fp16/bf16 training
fp16 = False
bf16 = False

# Batch Size
PerDevice = 1
PerDeviceEval = 1

# Gradient Accumalation
GradAccumSteps = 1

# Gradient checkpoint
GradientCheckpoint = True

# Gradient norm
GradNorm = 0.1

# Initial learning rate
LearningRate = 2e-4
# Weight decay per step
WeightDecay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate scheduler
LrSchedulerType = "cosine"

# Max steps
MaxSteps = -1

# Warmup ratio for linear warmup
WarmupRatio = 0.03

# Grouping sentences
GroupByLength = True

# Save Checkpoint after every 'n' amount of steps
SaveSteps = 0

# Verbose after 'n' amount of steps
LoggingSteps = 25

# SFT
MaxSeqLength = 512
Packing = False
DeviceMap = {"": 0}

dataset = load_dataset(DatasetName, split='train')
dataset = dataset.select(range(10000))

CompDtype = getattr(torch, Bnb4bitQuantDType)

BnbConfig = BitsAndBytesConfig(
    load_in_4bit=Use4bit,
    bnb_4bit_quant_type=Bnb4bitQuantType,
    bnb_4bit_compute_dtype=CompDtype,
    bnb_4bit_use_double_quant=UseNestedQuant
)


model = AutoModelForCausalLM.from_pretrained(
    ModelName,
    quantization_config=BnbConfig,
    device_map=DeviceMap,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(ModelName, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

PeftConfig = LoraConfig(
    lora_alpha=LoraAlpha,
    lora_dropout=LoraDropout,
    r=LoraR,
    bias='none',
    task_type='CAUSAL_LM'
)

TrainArgs = SFTConfig(
    output_dir=OutputDir,
    num_train_epochs=NumTrainEpochs,
    per_device_train_batch_size=PerDevice,
    gradient_accumulation_steps=GradAccumSteps,
    optim=optim,
    save_steps=SaveSteps,
    logging_steps=LoggingSteps,
    learning_rate=LearningRate,
    weight_decay=WeightDecay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=GradNorm,
    max_steps=MaxSteps,
    warmup_ratio=WarmupRatio,
    group_by_length=GroupByLength,
    lr_scheduler_type=LrSchedulerType,
    report_to="tensorboard",
    max_seq_length=MaxSeqLength,
    dataset_text_field="text",
    packing=Packing
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=PeftConfig,
    args=TrainArgs,
    processing_class=tokenizer,
)

trainer.train()

trainer.model.save_pretrained(MyModelName)

model = AutoModelForCausalLM.from_pretrained(ModelName)

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(prompt)
#print(result[0]['generated_text'])
print(result)

print(type(result[0]['generated_text']))

# Empty VRAM
del model
del pipe
del trainer
import gc
gc.collect()
gc.collect()

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    ModelName,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=DeviceMap,
)
model = PeftModel.from_pretrained(base_model, MyModelName)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(ModelName, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

import locale
locale.getpreferredencoding = lambda: "UTF-8"

model.push_to_hub("N0v4123/llama-3.2-1B-chat-finetune", check_pr=True)

tokenizer.push_to_hub("N0v4123/llama-3.2-1B-chat-finetune",check_pr=True)