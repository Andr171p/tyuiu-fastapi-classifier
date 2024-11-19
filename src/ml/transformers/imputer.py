import pandas as pd
from typing import Any
from functools import singledispatchmethod


class Imputer:

    @singledispatchmethod
    def transform(self, X) -> pd.DataFrame:
        raise NotImplementedError(f"Method 'transform' not implemented for type {type(X)}")

    @transform.register
    def _(self, X: dict) -> pd.DataFrame:
        return pd.DataFrame(
            data=X,
            index=[0]
        )

    @transform.register
    def _(self, X: list) -> pd.DataFrame:
        return pd.DataFrame(X)
