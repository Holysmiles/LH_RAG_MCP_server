"""Provider-independent contracts for text-based language models.

The abstractions in this module keep the rest of the application independent
from provider SDKs. Concrete providers are responsible for translating the
generic chat messages into their own request and response formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


# Messages remain mapping-based so providers can support optional fields without
# forcing the core contract to depend on a specific vendor's message model.
ChatMessage = Mapping[str, Any]


class BaseLLM(ABC):
    """Define the minimal contract implemented by every text LLM provider."""

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Generate an assistant response for an ordered chat history.

        Args:
            messages: Provider-independent message mappings in conversation order.

        Returns:
            The generated assistant text.
        """
        raise NotImplementedError
