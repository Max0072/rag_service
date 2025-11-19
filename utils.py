from typing import List, Any
import json
import torch


def parse_json_or_jsonl(raw: bytes, encoding: str = 'utf-8') -> List[Any]:
    text = raw.decode(encoding).strip()
    if text.startswith('['):
        data = json.loads(text)
        if isinstance(data, list):
            return data
        else:
            raise ValueError("JSON content is not a list")
    items = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# get device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")