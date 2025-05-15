# Run this with DDP with "accelerate launch test_sft.py"
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from accelerate import PartialState
from transformers import TrainerCallback
# > 1. <= multi gpu
# <= 1. <= single gpu

# ml > 1 node. <= multi node

# # device_map="DDP" # for DDP and running with `accelerate launch test_sft.py`
# accelerate launch main.py

device_map="FSDP" # for DDP and running with `accelerate launch test_sft.py`
# device_map='auto' # for DP and running with `python test_sft.py`

if device_map == "DDP" or "FSDP":
    device_string = PartialState().process_index
    device_map={'':device_string}

# Load the dataset
dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")

# Load the model + tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "codellama/CodeLlama-34b-hf"
# model_name = "meta-llama/Llama-2-70b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir='',
    use_cache = False,
    # attn_implementation="flash_attention_2",
    # low_cpu_mem_usage=True
    # torch_dtype = getattr(torch, "bfloat16"),
    # torch_dtype= torch.float16 #torch.bfloat16 if script_args.bf16 else torch.float16 if script_args.fp16 else torch.float32,

    # device_map = device_map,
)

# PEFT config
lora_alpha = 8
lora_dropout = 0.1
lora_r = 32
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
)

# Args 
max_seq_length = 512
output_dir = "/app/data/checkpoint"
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
optim = "adamw_torch"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1 # Approx the size of guanaco at bs 8, ga 2, 2 GPUs. 
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
    num_train_epochs=10,
    output_dir=output_dir,
    logging_dir= '/app/data/logs',
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = {"use_reentrant": True},
    report_to="tensorboard",
)
class TrainOnStartCallback(TrainerCallback):
                                def on_train_begin(self, args, state, control, logs=None, **kwargs):
                                    # Log training loss at step 0
                                    logs = logs or {}
                                    self.log(logs)

                                def log(self, logs):
                                    print(f"Logging at start: {logs}")
# Trainer 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    callbacks=[TrainOnStartCallback()]
)

# handle PEFT+FSDP case
# trainer.accelerator.model.print_trainable_parameters()
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

# Train
trainer.train()
# Define the save and push paths
new_model_repo = f"tonyshark/llama3"
local_save_path_model = f"{new_model_repo}-local"

trainer.save_model(local_save_path_model)
tokenizer.save_pretrained(local_save_path_model)

trainer.push_to_hub()
# free the memory again
del model
del trainer
torch.cuda.empty_cache()