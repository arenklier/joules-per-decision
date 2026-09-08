"""Why did the CoT multiplier collapse to 1.0 on NetArena?

Identical token counts under base and CoT are consistent with the models
ignoring the CoT instruction, but "consistent with" is not evidence. This
prints the actual generations side by side so the explanation can be checked
rather than assumed: if the CoT output contains no prose reasoning and is
essentially the same code block, the format constraint in NetArena's base
prompt is overriding the instruction.
"""
import json, os, difflib
os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

P = json.load(open("netarena_prompts.json", encoding="utf-8"))
TASKS = json.load(open("netarena_sample.json", encoding="utf-8"))
REPO = os.environ.get("REPO", "Qwen/Qwen2.5-14B-Instruct")
N = int(os.environ.get("N", 3))


def build(q, cot):
    parts = [P["BASE_PROMPT"]]
    if cot:
        parts.append(P["COT_PROMPT"])
    parts.append(P["PROMPT_SUFFIX"].replace("{input}", q))
    return "\n".join(parts)


def gen(model, tok, prompt, cap=768):
    enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)],
              return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        o = model.generate(**enc, max_new_tokens=cap)
    return tok.decode(o[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)


def main():
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    cfg = model.generation_config
    cfg.do_sample = False
    for k in ("temperature", "top_p", "top_k"):
        setattr(cfg, k, None)
    if cfg.pad_token_id is None:
        cfg.pad_token_id = tok.eos_token_id

    for t in TASKS[:N]:
        b = gen(model, tok, build(t["question"], False))
        c = gen(model, tok, build(t["question"], True))
        same = b.strip() == c.strip()
        ratio = difflib.SequenceMatcher(None, b, c).ratio()
        # prose outside the code fence is what a real CoT would add
        prose_b = len(b.split("```")[0].strip())
        prose_c = len(c.split("```")[0].strip())
        print(f"\n=== {t['id']} [{t['label']}] ===")
        print(f"  base {len(b):5d} chars | cot {len(c):5d} chars | identical={same} "
              f"| similarity={ratio:.3f}")
        print(f"  prose before code fence: base {prose_b} chars, cot {prose_c} chars")
        print(f"  --- cot output, first 320 chars ---")
        print("  " + c[:320].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
