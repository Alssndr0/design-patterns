"""
Template Method Pattern in Machine Learning/AI Pipelines

The Template Method design pattern defines the skeleton of an algorithm in a base class,
deferring some steps to subclasses. This allows you to customize parts of a pipeline while
keeping the overall process structure consistent.

In ML/AI, this is useful for:
- Standardizing the structure of pipelines (preprocessing, inference, postprocessing)
- Allowing customization of individual steps (e.g., custom cleaning, model logic, or formatting)
- Ensuring best practices and pipeline order are followed, while allowing extensions

In this example:
- `SentimentPipelineTemplate` defines the template method (`run_pipeline`) for text analysis.
- Subclasses override steps like `preprocess` or `postprocess` to customize behavior.
- The main workflow remains fixed, supporting consistency and reuse across many pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Template base class ---
class SentimentPipelineTemplate(ABC):
    def run_pipeline(self, text: str) -> Dict[str, Any]:
        print("[SentimentPipelineTemplate] Starting pipeline...")
        preprocessed = self.preprocess(text)
        result = self.infer(preprocessed)
        postprocessed = self.postprocess(result)
        print("[SentimentPipelineTemplate] Pipeline complete.")
        return postprocessed

    def preprocess(self, text: str) -> str:
        # Default: strip whitespace
        print(f"[SentimentPipelineTemplate] Preprocessing: '{text}'")
        return text.strip()

    @abstractmethod
    def infer(self, text: str) -> Dict[str, Any]:
        pass

    def postprocess(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Default: no-op
        print(f"[SentimentPipelineTemplate] Postprocessing: {result}")
        return result


# --- Concrete Template: Default sentiment analysis ---
class DefaultSentimentPipeline(SentimentPipelineTemplate):
    def __init__(self) -> None:
        print("[DefaultSentimentPipeline] Loading model...")
        self.pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        print("[DefaultSentimentPipeline] Model loaded.")

    def infer(self, text: str) -> Dict[str, Any]:
        print(f"[DefaultSentimentPipeline] Inference on: '{text}'")
        return self.pipeline(text)[0]


# --- Custom Template: With advanced preprocessing and custom postprocessing ---
class AdvancedSentimentPipeline(SentimentPipelineTemplate):
    def __init__(self) -> None:
        print("[AdvancedSentimentPipeline] Loading model...")
        self.pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        print("[AdvancedSentimentPipeline] Model loaded.")

    def preprocess(self, text: str) -> str:
        cleaned = text.lower().strip()
        print(
            f"[AdvancedSentimentPipeline] Custom preprocessing: '{text}' -> '{cleaned}'"
        )
        return cleaned

    def infer(self, text: str) -> Dict[str, Any]:
        print(f"[AdvancedSentimentPipeline] Inference on: '{text}'")
        return self.pipeline(text)[0]

    def postprocess(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Add confidence as percent
        formatted = {
            "sentiment": result["label"].capitalize(),
            "confidence_percent": f"{result['score'] * 100:.1f}%",
        }
        print(f"[AdvancedSentimentPipeline] Custom postprocessing: {formatted}")
        return formatted


if __name__ == "__main__":
    print("\n--- Template Method: Default Pipeline ---")
    default_pipeline = DefaultSentimentPipeline()
    res1 = default_pipeline.run_pipeline("  This is a great product!  ")
    print(f"Default Pipeline Result: {res1}")

    print("\n--- Template Method: Advanced Pipeline ---")
    advanced_pipeline = AdvancedSentimentPipeline()
    res2 = advanced_pipeline.run_pipeline("  This is a GREAT product!  ")
    print(f"Advanced Pipeline Result: {res2}")
