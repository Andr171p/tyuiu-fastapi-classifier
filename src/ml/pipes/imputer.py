import pandas as pd
from typing import Any, Dict, List, Optional
from functools import singledispatchmethod


class Imputer:
    def __init__(self, features: List[str]) -> None:
        self._features = features

    @singledispatchmethod
    def transform(self, arg: Any) -> Optional[pd.DataFrame, None]:
        raise ValueError(f"Cannot transform value of type {type(arg)}")

    @transform.register
    def _(self, X: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(
            data=X,
            index=[0]
        )

    @transform.register
    def _(self, X: List[Dict[str, Any]]) -> pd.DataFrame | None:
        return pd.DataFrame(X)


imp = Imputer(features=...)
