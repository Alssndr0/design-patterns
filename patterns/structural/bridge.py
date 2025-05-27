"""
Bridge Pattern in Machine Learning/AI Model Abstraction

The Bridge design pattern decouples an abstraction from its implementation, so the two can vary independently.
This is especially useful in ML/AI contexts when you want to:
- Separate the way a model is used from the specifics of the underlying backend (e.g., Hugging Face, ONNX, remote API, etc.)
- Allow new backends or usage patterns to be added with minimal code changes.

In this example:
- The `SentimentService` abstraction defines the high-level logic for analyzing text sentiment.
- The implementation (`ModelBackend`) is decoupled and can be swapped (Hugging Face, ONNX, or even a remote REST API).
- The two hierarchies (abstractions and implementations) can evolve independently, making the system highly extensible and maintainable.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Implementation hierarchy: ModelBackend ---
class ModelBackend(ABC):
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        pass


class HuggingFaceBackend(ModelBackend):
    def __init__(
        self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> None:
        print(f"[HuggingFaceBackend] Loading model '{model_name}'...")
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[HuggingFaceBackend] Model loaded.")

    def predict(self, text: str) -> Dict[str, Any]:
        print(f"[HuggingFaceBackend] Predicting: '{text}'")
        return self.pipeline(text)[0]


class ONNXBackend(ModelBackend):
    """
    Simulates an ONNX runtime backend. For demo, just returns a mock result.
    """

    def __init__(self) -> None:
        print("[ONNXBackend] ONNX model loaded.")

    def predict(self, text: str) -> Dict[str, Any]:
        print(f"[ONNXBackend] Predicting: '{text}'")
        # Dummy logic for demonstration
        score = 0.95 if "wonderful" in text else 0.2
        label = "POSITIVE" if score > 0.5 else "NEGATIVE"
        return {"label": label, "score": score}


# --- Abstraction hierarchy: SentimentService ---
class SentimentService:
    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        print("[SentimentService] Analyzing sentiment...")
        return self.backend.predict(text)


# --- Refined Abstraction: LoggingSentimentService ---
class LoggingSentimentService(SentimentService):
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        print("[LoggingSentimentService] Logging input text...")
        result = self.backend.predict(text)
        print(f"[LoggingSentimentService] Result: {result}")
        return result


if __name__ == "__main__":
    print("\n--- Using Hugging Face backend ---")
    hf_backend = HuggingFaceBackend()
    service1 = SentimentService(hf_backend)
    result1 = service1.analyze_sentiment("This is not a good movie!")
    print(f"Service1 Result: {result1}")

    print("\n--- Using ONNX backend ---")
    onnx_backend = ONNXBackend()
    service2 = SentimentService(onnx_backend)
    result2 = service2.analyze_sentiment("This is not a good movie!")
    print(f"Service2 Result: {result2}")

    print("\n--- Using LoggingSentimentService with Hugging Face backend ---")
    service3 = LoggingSentimentService(hf_backend)
    result3 = service3.analyze_sentiment("This is not a good good.")
    print(f"Service3 Result: {result3}")
