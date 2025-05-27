"""
Adapter Pattern in Machine Learning/AI Integration

The Adapter design pattern allows objects with incompatible interfaces to work together.
It acts as a "wrapper" that translates the interface of one class into an interface expected by the client.

In ML/AI engineering, this is useful when you want to use third-party, legacy, or external models that do not provide the same API as your other models.
By writing an adapter, you can make such models interchangeable with your standard code, enabling flexible integration and reducing code changes.

In this example, we demonstrate:
- A standard Hugging Face sentiment pipeline with a `.analyze(text)` method.
- An external/legacy sentiment model with a different API (`analyze_sentiment(text: str) -> Dict`).
- An Adapter class that wraps the legacy model and exposes a compatible `.analyze(text)` method,
  so both models can be used interchangeably in your workflow.
"""

from typing import Any, Dict

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Target interface expected by client code ---
class SentimentAnalyzer:
    def analyze(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError()


# --- Standard Hugging Face sentiment analyzer ---
class HuggingFaceSentimentAnalyzer(SentimentAnalyzer):
    def __init__(
        self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> None:
        print(f"[HuggingFaceSentimentAnalyzer] Loading model '{model_name}'...")
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[HuggingFaceSentimentAnalyzer] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[HuggingFaceSentimentAnalyzer] Analyzing: '{text}'")
        return self.pipeline(text)[0]


# --- External/legacy model with a different interface ---
class LegacySentimentModel:
    """
    Simulates a legacy model with a different API.
    """

    def __init__(self) -> None:
        print("[LegacySentimentModel] Model initialized.")

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        # Dummy logic for demo purposes
        score = 0.9 if "love" in text else 0.1
        label = "POSITIVE" if score > 0.5 else "NEGATIVE"
        print(
            f"[LegacySentimentModel] Analyzing: '{text}' -> label: {label}, score: {score}"
        )
        return {"label": label, "score": score}


# --- Adapter to make LegacySentimentModel compatible with SentimentAnalyzer interface ---
class LegacySentimentAdapter(SentimentAnalyzer):
    def __init__(self, legacy_model: LegacySentimentModel) -> None:
        self.legacy_model = legacy_model

    def analyze(self, text: str) -> Dict[str, Any]:
        # Call the legacy API and return its result as expected by the client
        return self.legacy_model.analyze_sentiment(text)


if __name__ == "__main__":
    print("\n--- Using standard Hugging Face analyzer ---")
    hf_analyzer: SentimentAnalyzer = HuggingFaceSentimentAnalyzer()
    result1: Dict[str, Any] = hf_analyzer.analyze("I love using this product!")
    print(f"Result 1: {result1}")

    print("\n--- Using legacy model with adapter ---")
    legacy_model = LegacySentimentModel()
    adapter: SentimentAnalyzer = LegacySentimentAdapter(legacy_model)
    result2: Dict[str, Any] = adapter.analyze("I love using this product!")
    print(f"Result 2: {result2}")

    print("\n--- Both analyzers now share a unified interface ---")
    for analyzer in [hf_analyzer, adapter]:
        result = analyzer.analyze("I don't like this at all.")
        print(f"Result: {result}")
