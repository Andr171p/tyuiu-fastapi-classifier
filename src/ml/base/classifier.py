from abc import ABC, abstractmethod

from typing import Any, Self


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X: Any, y: Any) -> Self: pass

    @abstractmethod
    def predict_proba(self, X: Any) -> Any: pass
