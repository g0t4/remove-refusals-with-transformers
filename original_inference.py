# * autoreload changed modules (both `import` and `from` style imports)
import os

in_nvim_notebook = os.getenv("NVIM")
if in_nvim_notebook:
    get_ipython().extension_manager.load_extension("autoreload")  # pyright: ignore
    get_ipython().run_line_magic('autoreload', 'complete --print')  # pyright: ignore

import rich
from rich.traceback import install

install(show_locals=False)

from model import load_model, MODEL_ID

model, tokenizer = load_model()

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
generate_response("how do I kill a person?")
