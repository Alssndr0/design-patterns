"""
Decorator Pattern in Machine Learning/AI Model Pipelines

The Decorator pattern allows you to add new functionality to objects dynamically, without modifying their code.
It is especially useful in ML/AI for adding preprocessing, postprocessing, logging, or performance tracking
to model inference pipelines, while keeping each concern modular and reusable.

In this example:
- The `SentimentAnalyzer` class provides basic sentiment analysis.
- We create decorators for logging, timing, and preprocessing (lowercasing).
- You can wrap the analyzer with one or more decorators to extend its behavior, in any order.

This lets you flexibly compose pipeline functionality without changing the original model code.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Base Component Interface ---
class SentimentAnalyzerBase(ABC):
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        pass


# --- Concrete Component ---
class SentimentAnalyzer(SentimentAnalyzerBase):
    def __init__(
        self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> None:
        print(f"[SentimentAnalyzer] Loading model '{model_name}'...")
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentAnalyzer] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[SentimentAnalyzer] Analyzing: '{text}'")
        return self.pipeline(text)[0]


# --- Decorator Base Class ---
class AnalyzerDecorator(SentimentAnalyzerBase):
    def __init__(self, analyzer: SentimentAnalyzerBase) -> None:
        self.analyzer = analyzer

    def analyze(self, text: str) -> Dict[str, Any]:
        return self.analyzer.analyze(text)


# --- Concrete Decorators ---
class LoggingDecorator(AnalyzerDecorator):
    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[LoggingDecorator] Input text: '{text}'")
        result = self.analyzer.analyze(text)
        print(f"[LoggingDecorator] Output: {result}")
        return result


class TimingDecorator(AnalyzerDecorator):
    def analyze(self, text: str) -> Dict[str, Any]:
        start = time.time()
        result = self.analyzer.analyze(text)
        duration = time.time() - start
        print(f"[TimingDecorator] Inference took {duration:.4f} seconds")
        return result


class LowercaseDecorator(AnalyzerDecorator):
    def analyze(self, text: str) -> Dict[str, Any]:
        new_text = text.lower()
        print(f"[LowercaseDecorator] Lowercased text: '{new_text}'")
        return self.analyzer.analyze(new_text)


if __name__ == "__main__":
    print("\n--- Plain analyzer ---")
    analyzer = SentimentAnalyzer()
    result1 = analyzer.analyze("This product is EXCELLENT!")
    print(f"Plain Result: {result1}")

    print("\n--- Analyzer with logging and timing ---")
    logged = LoggingDecorator(analyzer)
    timed_logged = TimingDecorator(logged)
    result2 = timed_logged.analyze("This product is EXCELLENT!")
    print(f"Timed + Logged Result: {result2}")

    print("\n--- Analyzer with lowercasing, logging, and timing ---")
    lowercased = LowercaseDecorator(analyzer)
    decorated = LoggingDecorator(TimingDecorator(lowercased))
    result3 = decorated.analyze("THIS PRODUCT IS EXCELLENT!")
    print(f"Fully Decorated Result: {result3}")
