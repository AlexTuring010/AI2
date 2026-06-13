import hashlib
import pickle
from pathlib import Path
from typing import Any


def _hash_key(key: tuple) -> str:
    raw = str(key).encode()
    return hashlib.md5(raw).hexdigest()[:12]


def save_artifact(obj: Any, cache_dir: str, name: str) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / name
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_artifact(cache_dir: str, name: str) -> Any:
    path = Path(cache_dir) / name
    with open(path, "rb") as f:
        return pickle.load(f)


def artifact_exists(cache_dir: str, name: str) -> bool:
    return (Path(cache_dir) / name).exists()


def token_cache_name(model_name: str, max_length: int, input_fmt: str, split_id: int) -> str:
    key = (model_name, max_length, input_fmt, split_id)
    return f"tokens_{_hash_key(key)}.pkl"
