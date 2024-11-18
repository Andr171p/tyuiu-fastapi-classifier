import dill
import pickle
from typing import Any
from pathlib import Path


def load_pkl(filename: Path) -> Any:
    with open(
        file=filename,
        mode='rb'
    ) as file:
        obj = pickle.load(file)
    return obj


def load_dill(filename: Path) -> Any:
    with open(
        file=filename,
        mode='rb'
    ) as file:
        obj = dill.load(file)
    return obj
