import pytest
from task_arithmetic.data import language_model_pairs
from task_arithmetic.config import prompt


class DummyTokenizer:
    eos_token_id=0
    def __call__(self, text, **kwargs):
        return {"input_ids":[i+1 for i,_ in enumerate(text.split())]}
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if i)


def test_lm_uses_text_only():
    rows=[{"text":" ".join(["token"]*100)} for _ in range(3)]
    pairs=language_model_pairs(rows,DummyTokenizer(),chunk_tokens=16)
    assert pairs and all(x["input"].startswith("continue: ") and x["target"] for x in pairs)
    assert all("label" not in x for x in pairs)


def test_prompt_prefix():
    assert prompt("amazon_sent","hello")=="sentiment: hello"
    assert prompt("agnews_topic","hello")=="topic: hello"
