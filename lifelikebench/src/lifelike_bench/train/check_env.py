"""Check that the training stack is importable and a GPU is visible.

  python -m lifelike_bench.train.check_env [--model PATH]   (with --model: also loads the tokenizer
                                                             and renders one chat turn)
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    a = ap.parse_args()
    missing = []
    for m in ("torch", "transformers", "peft"):
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        print("MISSING:", " ".join(missing))
        print("install on the login node:  pip install " + " ".join(missing))
        sys.exit(1)
    from .common import env_report, prompt_and_target
    print(json.dumps(env_report(), indent=1))
    if a.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        p, t = prompt_and_target(tok, "你是林晚。", [{"role": "user", "content": "嗨"}], "你好呀。")
        print("--- prompt tail ---\n" + p[-120:] + "\n--- target ---\n" + repr(t))
    print("OK")


if __name__ == "__main__":
    main()
