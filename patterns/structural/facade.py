"""
Facade Pattern in Machine Learning/AI Pipelines

The Facade design pattern provides a simplified, high-level interface to a set of complex subsystems.
It makes a library, framework, or workflow easier to use by hiding internal complexity and offering
a single entry point.

In ML/AI, this is useful when you want to bundle multiple models, preprocessing, postprocessing,
and data cleaning steps behind a single, user-friendly API. The user interacts with the facade,
which takes care of calling the right components in the right order.

In this example:
- Several pipeline steps (preprocessing, sentiment analysis, and postprocessing) are defined as separate classes.
- The `SentimentAnalysisFacade` class hides all the details and exposes a simple `.analyze(text)` method.
- The client only needs to interact with the facade, making complex workflows easy to use and maintain.
"""

from typing import Any, Dict

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Subsystem 1: Preprocessing ---
class Preprocessor:
    def process(self, text: str) -> str:
        cleaned = text.strip()
        print(f"[Preprocessor] Cleaned text: '{cleaned}'")
        return cleaned


# --- Subsystem 2: Sentiment Analysis Model ---
class SentimentModel:
    def __init__(
        self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> None:
        print(f"[SentimentModel] Loading model '{model_name}'...")
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentModel] Model loaded.")

    def predict(self, text: str) -> Dict[str, Any]:
        print("[SentimentModel] Predicting sentiment...")
        result = self.pipeline(text)[0]
        print(f"[SentimentModel] Raw result: {result}")
        return result


# --- Subsystem 3: Postprocessing ---
class Postprocessor:
    def process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Example: convert score to percentage, pretty format
        formatted = {
            "label": result["label"].capitalize(),
            "confidence_percent": f"{result['score'] * 100:.1f}%",
        }
        print(f"[Postprocessor] Formatted result: {formatted}")
        return formatted


# --- Facade: User-facing interface ---
class SentimentAnalysisFacade:
    def __init__(self) -> None:
        self.preprocessor = Preprocessor()
        self.model = SentimentModel()
        self.postprocessor = Postprocessor()

    def analyze(self, text: str) -> Dict[str, Any]:
        print("[SentimentAnalysisFacade] Starting full analysis pipeline...")
        cleaned_text = self.preprocessor.process(text)
        raw_result = self.model.predict(cleaned_text)
        formatted = self.postprocessor.process(raw_result)
        print("[SentimentAnalysisFacade] Pipeline complete.")
        return formatted


if __name__ == "__main__":
    facade = SentimentAnalysisFacade()
    print("\n--- Facade usage example ---")
    result = facade.analyze("   I absolutely love this product!   ")
    print(f"Final Facade Result: {result}")
