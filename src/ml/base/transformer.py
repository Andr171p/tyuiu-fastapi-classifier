from abc import ABC, abstractmethod

from typing import Any, Dict, List, Union


class BaseTransformer(ABC):
    @abstractmethod
    def fit_transform(
            self,
            X: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> Any:
        pass

    @abstractmethod
    def transform(
            self,
            X: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> Any:
        pass
