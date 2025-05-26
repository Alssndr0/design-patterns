"""
Singleton Pattern in Machine Learning/AI Model Serving

The Singleton design pattern ensures that a class has only one instance and provides a global point of access to that instance.
This is particularly useful in machine learning and AI engineering contexts where you often work with heavy, resource-intensive objects
such as pre-trained models or inference pipelines.

Loading a machine learning model can consume a significant amount of memory and computational resources.
If you are building an application (for example, an API service) that handles many requests,
you want to avoid reloading the model for every request or for every new instance of your handler class.
The Singleton pattern solves this by guaranteeing that only a single instance of the model loader exists across your application,
regardless of how many times you instantiate the handler class.

In this example, we use the Singleton pattern (via a metaclass) to ensure that only one instance of a Hugging Face sentiment analysis pipeline
is ever loaded. No matter how many times the SentimentAnalyzer class is instantiated, the Hugging Face pipeline is only initialized once,
saving both memory and time, and providing consistent predictions across your application.
"""

from typing import Any, Dict, Type, TypeVar

from transformers import pipeline
from transformers.pipelines.base import Pipeline

T = TypeVar("T")


class Singleton(type):
    """
    Singleton Metaclass: Ensures only one instance of a class exists.
    """

    _instances: Dict[Type, Any] = {}

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in cls._instances:
            print(f"[Singleton] Creating a new instance of {cls.__name__}")
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            print(f"[Singleton] Returning existing instance of {cls.__name__}")
        return cls._instances[cls]


class SentimentAnalyzer(metaclass=Singleton):
    def __init__(self) -> None:
        print(
            "[SentimentAnalyzer] Loading sentiment analysis pipeline (this is heavy, only happens once)..."
        )
        self.pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        print("[SentimentAnalyzer] Pipeline loaded.")

    def analyze(self, text: str) -> Any:
        print(f"[SentimentAnalyzer] Analyzing text: '{text}'")
        return self.pipeline(text)


if __name__ == "__main__":
    print("\n--- Creating first SentimentAnalyzer instance ---")
    analyzer1: SentimentAnalyzer = SentimentAnalyzer()
    result1: Any = analyzer1.analyze("I love using this product! It's amazing.")
    print(f"Result 1: {result1}")

    print(
        "\n--- Creating second SentimentAnalyzer instance (should reuse the first) ---"
    )
    analyzer2: SentimentAnalyzer = SentimentAnalyzer()
    result2: Any = analyzer2.analyze("I am very disappointed and sad.")
    print(f"Result 2: {result2}")

    print("\n--- Comparing instances ---")
    print(f"analyzer1 is analyzer2: {analyzer1 is analyzer2}")
    print(f"analyzer1.pipeline id: {id(analyzer1.pipeline)}")
    print(f"analyzer2.pipeline id: {id(analyzer2.pipeline)}")
