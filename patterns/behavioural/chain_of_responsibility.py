"""
Chain of Responsibility Pattern in Machine Learning/AI Inference Pipelines

The Chain of Responsibility pattern passes a request along a chain of handlers.
Each handler decides either to process the request or to pass it to the next handler in the chain.
This is especially useful in ML/AI for building inference or data processing pipelines
where steps like filtering, pre/postprocessing, model routing, or fallback logic can be chained.

In this example:
- Each handler in the chain performs a check or processing step (e.g., empty input, profanity filter, sentiment analysis).
- If a handler cannot process the input, it passes it to the next handler.
- The chain is flexible and handlers can be composed in any order.

This design makes it easy to add, remove, or reorder processing steps, and keeps each step modular.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Handler interface ---
class Handler(ABC):
    def __init__(self, next_handler: Optional["Handler"] = None) -> None:
        self.next_handler = next_handler

    @abstractmethod
    def handle(self, text: str) -> Dict[str, Any]:
        pass

    def set_next(self, handler: "Handler") -> "Handler":
        self.next_handler = handler
        return handler


# --- Concrete Handlers ---
class EmptyInputHandler(Handler):
    def handle(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            print("[EmptyInputHandler] Input is empty. Aborting.")
            return {"error": "Input text is empty."}
        print("[EmptyInputHandler] Input is non-empty. Passing to next handler.")
        return self.next_handler.handle(text) if self.next_handler else {}


class ProfanityFilterHandler(Handler):
    PROFANITIES = {"damn", "hell", "shit"}  # Demo list

    def handle(self, text: str) -> Dict[str, Any]:
        if any(bad_word in text.lower() for bad_word in self.PROFANITIES):
            print("[ProfanityFilterHandler] Profanity detected! Aborting.")
            return {"error": "Profanity detected in input."}
        print("[ProfanityFilterHandler] No profanity. Passing to next handler.")
        return self.next_handler.handle(text) if self.next_handler else {}


class SentimentAnalysisHandler(Handler):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        next_handler: Optional[Handler] = None,
    ) -> None:
        super().__init__(next_handler)
        print(f"[SentimentAnalysisHandler] Loading model '{model_name}'...")
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentAnalysisHandler] Model loaded.")

    def handle(self, text: str) -> Dict[str, Any]:
        print("[SentimentAnalysisHandler] Analyzing sentiment...")
        result = self.pipeline(text)[0]
        print(f"[SentimentAnalysisHandler] Result: {result}")
        return result


if __name__ == "__main__":
    # Build the chain: EmptyInput -> ProfanityFilter -> SentimentAnalysis
    empty_handler = EmptyInputHandler()
    profanity_handler = ProfanityFilterHandler()
    sentiment_handler = SentimentAnalysisHandler()
    empty_handler.set_next(profanity_handler).set_next(sentiment_handler)

    chain = empty_handler

    print("\n--- Chain of Responsibility Usage Examples ---")
    test_cases = ["", "This product is the damn best.", "This is absolutely fantastic!"]
    for text in test_cases:
        print(f"\nInput: '{text}'")
        result = chain.handle(text)
        print(f"Chain Result: {result}")
