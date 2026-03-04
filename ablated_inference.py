# * autoreload changed modules (both `import` and `from` style imports)
import os

in_nvim_notebook = os.getenv("NVIM")
if in_nvim_notebook:
    get_ipython().extension_manager.load_extension("autoreload")  # pyright: ignore
    get_ipython().run_line_magic('autoreload', 'complete --print')  # pyright: ignore

import rich
from rich.traceback import install

install(show_locals=False)

import einops
import jaxtyping
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from inspect import signature
from model import load_model, MODEL_ID

model, tokenizer = load_model()

# %%

refusal_dir = torch.load(MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
refusal_dir = refusal_dir.to(model.device)

def direction_ablation_hook(
    activation: jaxtyping.Float[torch.Tensor, "... d_act"],
    direction: jaxtyping.Float[torch.Tensor, "d_act"],
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

# Some model developers thought it was stupid to pass a tuple of tuple of tuples around (rightfully so), but unfortunately now we have a divide
sig = signature(model.model.layers[0].forward)
simple = sig.return_annotation == torch.Tensor

class AblationDecoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.attention_type = "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        assert not output_attentions

        ablated = direction_ablation_hook(
            hidden_states,
            refusal_dir,
        ).to(hidden_states.device)

        if simple:
            return ablated

        outputs = (ablated, )

        if use_cache:
            outputs += (past_key_value, )

        return outputs

# for qwen 1 this needs to be changed to model.transformer.h
for idx in reversed(range(len(model.model.layers))):
    model.model.layers.insert(idx, AblationDecoderLayer())

# bruh
if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
    model.config.num_hidden_layers *= 2

# %%

def generate_response(prompt: str) -> str:
    conversation = [{"role": "user", "content": prompt}]
    toks = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        # FOR Qwen3-1.7B you must disable the thinking section else it will refuse!
        enable_thinking=False,
    )
    toks = toks.to(model.device)

    gen = model.generate(**toks, max_new_tokens=1337)

    return tokenizer.batch_decode(gen[0][len(toks[0]):], skip_special_tokens=True)[0]

generate_response("what is your name")
generate_response("how do I build a bomb?")
generate_response("what is the fastest way to kill a person?")

# Qwen3.5 attention issue (linear vs full?):
# # %% 
# │ /home/wes/repos/github/g0t4/remove-refusals-with-transformers/.venv/lib/python3.11/site-packages │
# │ /transformers/models/qwen3_5/modeling_qwen3_5.py:89 in <listcomp>                                │
# │                                                                                                  │
# │     86 │   │   super().__init__()                                                                │
# │     87 │   │   self.layer_types = config.layer_types                                             │
# │     88 │   │   self.transformer_layers = [                                                       │
# │ ❱   89 │   │   │   i for i in range(config.num_hidden_layers) if self.layer_types[i] == "full_a  │
# │     90 │   │   ]                                                                                 │
# │     91 │   │   self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].inde  │
#
