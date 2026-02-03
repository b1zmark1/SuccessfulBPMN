from __future__ import annotations

from abc import ABC, abstractmethod


class BaseNarratorProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

