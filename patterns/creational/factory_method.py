"""
Factory Method Pattern in Machine Learning/AI Model Selection

The Factory Method pattern defines an interface for creating an object, but lets subclasses decide which class to instantiate.
This allows you to defer instantiation to subclasses or child classes, providing flexibility in object creation.

In ML/AI engineering, the Factory Method is useful when you need to create models or pipelines dynamically based on
runtime input or configuration, but want to encapsulate the selection logic away from the client code.

In this example, the Factory Method is used to select between different types of sentiment analyzers
(a standard English sentiment model, a multilingual model, or a fast/compact model).
This enables easily swapping or extending model types by simply adding a new subclass,
without modifying the factory or main workflow logic.

The client code simply requests a sentiment analyzer of a given type; the factory method handles all details of object creation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Product Interface ---
class SentimentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        pass


# --- Concrete Products ---
class EnglishSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self) -> None:
        print("[EnglishSentimentAnalyzer] Loading English sentiment model...")
        self.pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        print("[EnglishSentimentAnalyzer] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[EnglishSentimentAnalyzer] Analyzing: '{text}'")
        return self.pipeline(text)[0]


class MultilingualSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self) -> None:
        print("[MultilingualSentimentAnalyzer] Loading multilingual sentiment model...")
        self.pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
        )
        print("[MultilingualSentimentAnalyzer] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[MultilingualSentimentAnalyzer] Analyzing: '{text}'")
        return self.pipeline(text)[0]


class FastSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self) -> None:
        print("[FastSentimentAnalyzer] Loading fast, compact sentiment model...")
        self.pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
        )
        print("[FastSentimentAnalyzer] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[FastSentimentAnalyzer] Analyzing: '{text}'")
        return self.pipeline(text)[0]


# --- Factory Method Base Class ---
class SentimentAnalyzerFactory(ABC):
    @abstractmethod
    def create_analyzer(self) -> SentimentAnalyzer:
        pass


# --- Concrete Factories ---
class EnglishAnalyzerFactory(SentimentAnalyzerFactory):
    def create_analyzer(self) -> SentimentAnalyzer:
        return EnglishSentimentAnalyzer()


class MultilingualAnalyzerFactory(SentimentAnalyzerFactory):
    def create_analyzer(self) -> SentimentAnalyzer:
        return MultilingualSentimentAnalyzer()


class FastAnalyzerFactory(SentimentAnalyzerFactory):
    def create_analyzer(self) -> SentimentAnalyzer:
        return FastSentimentAnalyzer()


# --- Client code that selects the appropriate factory/method ---
def get_factory(analyzer_type: str) -> SentimentAnalyzerFactory:
    if analyzer_type == "english":
        return EnglishAnalyzerFactory()
    elif analyzer_type == "multilingual":
        return MultilingualAnalyzerFactory()
    elif analyzer_type == "fast":
        return FastAnalyzerFactory()
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")


if __name__ == "__main__":
    for analyzer_type in ["english", "multilingual", "fast"]:
        print(f"\n--- Using {analyzer_type.capitalize()} Sentiment Analyzer ---")
        factory: SentimentAnalyzerFactory = get_factory(analyzer_type)
        analyzer: SentimentAnalyzer = factory.create_analyzer()
        text = (
            "I love this product!"
            if analyzer_type != "multilingual"
            else "Me encanta este producto!"
        )
        result: Dict[str, Any] = analyzer.analyze(text)
        print(f"Result: {result}")
