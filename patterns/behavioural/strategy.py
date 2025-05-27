"""
Strategy Pattern in Machine Learning/AI Model Selection

The Strategy design pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable.
The client can select which strategy to use at runtime, allowing for flexible model selection, data processing,
or evaluation logic.

In ML/AI, this is useful for:
- Swapping out different models or inference backends dynamically
- Switching between preprocessing, postprocessing, or thresholding algorithms
- Parameterizing an API or workflow for flexible experiments

In this example:
- `SentimentAnalyzerContext` holds a reference to a `SentimentStrategy` and delegates inference to it.
- Different strategies implement the interface for various Hugging Face models.
- The strategy can be switched at runtime, changing the inference logic and model used.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Strategy interface ---
class SentimentStrategy(ABC):
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        pass


# --- Concrete Strategies ---
class EnglishSentimentStrategy(SentimentStrategy):
    def __init__(self) -> None:
        print("[EnglishSentimentStrategy] Loading English model...")
        self.pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        print("[EnglishSentimentStrategy] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print("[EnglishSentimentStrategy] Analyzing...")
        return self.pipeline(text)[0]


class MultilingualSentimentStrategy(SentimentStrategy):
    def __init__(self) -> None:
        print("[MultilingualSentimentStrategy] Loading multilingual model...")
        self.pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
        )
        print("[MultilingualSentimentStrategy] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print("[MultilingualSentimentStrategy] Analyzing...")
        return self.pipeline(text)[0]


class FastSentimentStrategy(SentimentStrategy):
    def __init__(self) -> None:
        print("[FastSentimentStrategy] Loading fast model...")
        self.pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
        )
        print("[FastSentimentStrategy] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print("[FastSentimentStrategy] Analyzing...")
        return self.pipeline(text)[0]


# --- Context: Holds and uses a strategy ---
class SentimentAnalyzerContext:
    def __init__(self, strategy: SentimentStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: SentimentStrategy) -> None:
        print(
            f"[SentimentAnalyzerContext] Switching strategy to: {strategy.__class__.__name__}"
        )
        self._strategy = strategy

    def analyze(self, text: str) -> Dict[str, Any]:
        return self._strategy.analyze(text)


if __name__ == "__main__":
    print("\n--- Strategy Pattern: Model Selection at Runtime ---")
    context = SentimentAnalyzerContext(EnglishSentimentStrategy())

    print("\nFirst inference (English strategy):")
    result1 = context.analyze("I love this product!")
    print(f"Result1: {result1}")

    print("\nSwitching to multilingual strategy:")
    context.set_strategy(MultilingualSentimentStrategy())
    result2 = context.analyze("Me encanta este producto!")
    print(f"Result2: {result2}")

    print("\nSwitching to fast model strategy:")
    context.set_strategy(FastSentimentStrategy())
    result3 = context.analyze("Awesome experience!")
    print(f"Result3: {result3}")
