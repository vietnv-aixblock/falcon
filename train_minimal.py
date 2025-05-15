import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainerCallback
import argparse
import wandb
wandb.login('allow',"69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")
# https://huggingface.co/blog/llama31
# https://blog.eleuther.ai/transformer-math/
# {
#   'token_length': '4096',
#   'accuracy': '70',
#   'sampling_frequency': '48000',
#   'mono': True,
#   'fps': '60',
#   'resolution': '320',
#   'image_width': '256',
#   'image_height': '256',
#   'framework': 'pytorch',
#   'precision': 'fp16',
#   "dataset_format": "YOLO",
#    "cuda": "11.4", 
#    "task": "",
#   'dataset_sample': [
#     {
#       "name": "",
#       "value": "Sujithanumala/Llama_3.2_1B_IT_dataset",
#       "desc": ""
#     }
#   ],
#   'weight': [
#     {
#       "name": "",
#       "value": "facebook/opt-125m",
#       "size": 11111,
#       "paramasters": 22222,
#       "tflops": 3333,
#         "nvlink": 3333
#     }
#   ],
#   'calculate_compute_gpu': {
#     "paramasters": 759808,
#     "mac": "2279424000000.0005",
#     "gpu_memory": 16106127360,
#     "tflops": 4558848000000.001,
#     "time": 0.00019444444444444443,
#     "total_cost": "0.159",
#     "total_power_consumption": 0,
#     "can_rent": true,
#     "token_symbol": "usd"
#   },
#   'estimate_time': 1,
#   'estimate_cost': '0.159',
#   "TrainingArguments": [
#     {
#       "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
#       "model_type": "LlamaForCausalLM",
#       "tokenizer_type": "AutoTokenizer",
#       "load_in_8bit": true,
#       "load_in_4bit": false,
#       "strict": false,
#       "chat_template": "llama3",
#       "rl": "dpo",
#       "datasets": [
#         {
#           "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
#           "type": "chat_template.default",
#           "field_messages": "conversation",
#           "field_chosen": "chosen",
#           "field_rejected": "rejected",
#           "message_field_role": "role",
#           "message_field_content": "content",
#           "roles": {
#             "system": [
#               "system"
#             ],
#             "user": [
#               "user"
#             ],
#             "assistant": [
#               "assistant"
#             ]
#           }
#         }
#       ],
#       "dataset_prepared_path": null,
#       "val_set_size": 0.05,
#       "output_dir": "./outputs/lora-out",
#       "sequence_len": 4096,
#       "sample_packing": false,
#       "pad_to_sequence_len": true,
#       "adapter": "lora",
#       "lora_model_dir": null,
#       "lora_r": 32,
#       "lora_alpha": 16,
#       "lora_dropout": 0.05,
#       "lora_target_linear": true,
#       "lora_fan_in_fan_out": null,
#       "wandb_project": null,
#       "wandb_entity": null,
#       "wandb_watch": null,
#       "wandb_name": null,
#       "wandb_log_model": null,
#       "gradient_accumulation_steps": 4,
#       "micro_batch_size": 2,
#       "num_epochs": 4,
#       "optimizer": "adamw_bnb_8bit",
#       "lr_scheduler": "cosine",
#       "learning_rate": 0.0002,
#       "train_on_inputs": false,
#       "group_by_length": false,
#       "bf16": "auto",
#       "fp16": null,
#       "tf32": false,
#       "gradient_checkpointing": true,
#       "early_stopping_patience": null,
#       "resume_from_checkpoint": null,
#       "local_rank": null,
#       "logging_steps": 1,
#       "xformers_attention": null,
#       "flash_attention": true,
#       "s2_attention": null,
#       "warmup_steps": 10,
#       "evals_per_epoch": 4,
#       "eval_table_size": null,
#       "eval_max_new_tokens": 128,
#       "saves_per_epoch": 1,
#       "debug": null,
#       "deepspeed": null,
#       "weight_decay": 0,
#       "fsdp": null,
#       "fsdp_config": null
#     }
#   ]
# }
# https://huggingface.co/docs/transformers/en/accelerate
# https://github.com/huggingface/trl/issues/527
# model_id = kwargs.get("model_id", "bigscience/bloomz-1b7")  #"tiiuae/falcon-7b" "bigscience/bloomz-1b7" `zanchat/falcon-1b` `appvoid/llama-3-1b` meta-llama/Llama-3.2-3B` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`
#                 dataset_id = kwargs.get("dataset_id","lucasmccabe-lmi/CodeAlpaca-20k") #gingdev/llama_vi_52k kigner/ruozhiba-llama3-tt
#                 num_train_epochs = kwargs.get("num_train_epochs", 3)
#               
#                 learning_rate = kwargs.get("learning_rate", 2e-4)
#                 bf16 = kwargs.get("bf16", False)
#                 fp16 = kwargs.get("fp16", False)
#                 use_cpu = kwargs.get("use_cpu", True)
#                 push_to_hub = kwargs.get("push_to_hub", False)
#                 hf_model_id = kwargs.get("hf_model_id", "tonyshark/llama3")
                
#                 max_seq_length = kwargs.get("max_seq_length", 1024)
#                 framework = kwargs.get("framework", "huggingface")


parser = argparse.ArgumentParser(description='AIxBlock')
parser.add_argument(
    '-m', '--model_id', dest='model_id', type=str, default="Qwen/Qwen1.5-1.8B-Chat",
    help='model id')
parser.add_argument(
    '-d', '--dataset_id', dest='dataset_id', type=str, default="argilla/dpo-mix-7k",
    help='dataset id')
parser.add_argument(
    '-n', '--num_train_epochs', dest='num_train_epochs', type=int, default=8,
    help='num_train_epochs')

parser.add_argument(
    '-bf', '--bf16', dest='bf16', type=str, default='',
    help='bf16')
parser.add_argument(
    '-fp', '--fp16', dest='fp16', type=str, default='',
    help='fp16')
parser.add_argument(
    '-ff', '--tf32', dest='tf32', type=str, default='',
    help='tf32')
parser.add_argument(
    '-c', '--use_cpu', dest='use_cpu', type=int, default=8,
    help='use_cpu')
parser.add_argument(
    '-ph', '--push_to_hub', dest='push_to_hub', type=str, default='',
    help='push_to_hub')
parser.add_argument(
    '-hf', '--hf_model_id', dest='hf_model_id', type=int, default=8,
    help='hf_model_id')
parser.add_argument(
    '-ms', '--max_seq_length', dest='max_seq_length', type=str, default='',
    help='max_seq_length')
parser.add_argument(
    '-f', '--framework', dest='framework', type=int, default=8,
    help='framework')
parser.add_argument(
    '-per_device_train_batch_size', '--per_device_train_batch_size', dest='per_device_train_batch_size', type=int, default=3,
    help='per_device_train_batch_size')
parser.add_argument(
    '-gradient_accumulation_steps', '--gradient_accumulation_steps', dest='gradient_accumulation_steps', type=int, default=16,
    help='gradient_accumulation_steps')
parser.add_argument(
    '-gradient_checkpointing', '--gradient_checkpointing', dest='gradient_checkpointing', type=bool, default=True,
    help='gradient_checkpointing')
parser.add_argument(
    '-optim', '--optim', dest='optim', type=str, default="adamw_torch",
    help='optim')
parser.add_argument(
    '-logging_steps', '--logging_steps', dest='logging_steps', type=int, default=3,
    help='logging_steps')
parser.add_argument(
    '-gradient_checkpointing', '--gradient_checkpointing', dest='gradient_checkpointing', type=int, default=10,
    help='gradient_checkpointing')
parser.add_argument(
    '-learning_rate', '--learning_rate', dest='learning_rate', type=float, default=2e-4,
    help='learning_rate')
parser.add_argument(
    '-max_grad_norm', '--max_grad_norm', dest='max_grad_norm', type=float, default=0.3,
    help='max_grad_norm')
parser.add_argument(
    '-lora_alpha', '--lora_alpha', dest='lora_alpha', type=float, default=8,
    help='lora_alpha')
parser.add_argument(
    '-lora_dropout', '--lora_dropout', dest='lora_dropout', type=float, default=0.05,
    help='lora_dropout')
parser.add_argument(
    '-bias', '--bias', dest='bias', type=float, default="none",
    help='bias')
parser.add_argument(
    '-target_modules', '--target_modules', dest='target_modules', type=str, default="all-linear",
    help='target_modules')
parser.add_argument(
    '-task_type', '--task_type', dest='task_type', type=str, default="CAUSAL_LM",
    help='task_type')

args = parser.parse_args()

print(args)

# args.framework = 'huggingface'
# args.bf16 = 'True'
# args.fp16 = 'True'
# args.use_cpu = 'True'
# args.push_to_hub = 'True'
# args.hf_model_id = 'tonyshark/llama3'
# args.max_seq_length = 1024
# args.num_train_epochs = 10
# args.dataset_id = 'lucasmccabe-lmi/CodeAlpaca-20k'
# args.model_id = 'Qwen/Qwen1.5-4B-Chat'

output_dir = '/app/data/checkpoint'

from_pretrained_kwargs = {
    # "low_cpu_mem_usage":True,
    "use_cache":True,
    # "torch_dtype": torch.float16 ,#getattr(torch, "bfloat16"),
    "trust_remote_code": True,
    "cache_dir": '',
    # "attn_implementation": "flash_attention_2",
    # "device_map": "auto",
}

model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    **from_pretrained_kwargs
)

tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

if '<unk>' in tokenizer.get_vocab():
    tokenizer.pad_token = '<unk>'
else:
    tokenizer.pad_token = tokenizer.eos_token

# Update pad token id in model and its config
model.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA configuration and application
use_lora = True
if use_lora:
    lora_config = LoraConfig(
        r=32,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules, #["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type=args.task_type,
        modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"]
    )

    model = get_peft_model(model, lora_config)

# Dataset loading
train_dataset = load_dataset(args.dataset_id, split='train[:100]')
train_dataset = train_dataset.rename_column('chosen', 'messages')

training_args = TrainingArguments(
    max_steps=10,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    bf16=args.bf16,
    fp16=args.fp16,
    learning_rate=args.learning_rate,
    lr_scheduler_type="constant",
    save_steps=0,
    optim=args.optim,
    save_strategy="steps",
    logging_dir= '/app/data/logs',
    output_dir=output_dir,
    warmup_ratio=0.03,
    logging_steps=args.logging_steps,
    hub_private_repo=True,
    gradient_checkpointing=args.gradient_checkpointing,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    save_total_limit=1,
    report_to="wandb",
)
class TrainOnStartCallback(TrainerCallback):
                                def on_train_begin(self, args, state, control, logs=None, **kwargs):
                                    # Log training loss at step 0
                                    logs = logs or {}
                                    self.log(logs)

                                def log(self, logs):
                                    print(f"Logging at start: {logs}")
trainer = SFTTrainer(
    max_seq_length=2048,
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[TrainOnStartCallback()]
)

# Training
model.config.use_cache = False
trainer.train()

# Define the save and push paths
new_model_repo = f"tonyshark/Qwen1.5-4B"
local_save_path_model = f"{new_model_repo}-local"

trainer.save_model(local_save_path_model)
tokenizer.save_pretrained(local_save_path_model)

trainer.push_to_hub()
# free the memory again
del model
del trainer
torch.cuda.empty_cache()