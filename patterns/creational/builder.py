"""
Builder Pattern in Machine Learning/AI Model Configuration

The Builder pattern separates the construction of a complex object from its representation,
allowing the same construction process to create different representations or configurations.

In ML/AI engineering, you may want to construct models or pipelines with various configurations
(e.g., model type, preprocessing steps, thresholds) without cluttering your code with many constructors
or conditional logic. The Builder pattern offers a flexible way to define these construction steps,
so you can assemble the model pipeline in a controlled and readable way.

In this example, we use the Builder pattern to assemble a sentiment analysis service with configurable preprocessing.
You can build different versions of the analyzer (e.g., with or without text lowercasing, with custom thresholds, etc.)
while keeping the construction process readable and reusable.
"""

from typing import Any, Dict, Optional

from transformers import pipeline
from transformers.pipelines.base import Pipeline


class SentimentAnalyzerBuilder:
    def __init__(self) -> None:
        self._use_lowercase: bool = False
        self._threshold: float = 0.5
        self._model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"

    def with_lowercase(self, use_lowercase: bool = True) -> "SentimentAnalyzerBuilder":
        self._use_lowercase = use_lowercase
        return self

    def with_threshold(self, threshold: float) -> "SentimentAnalyzerBuilder":
        self._threshold = threshold
        return self

    def with_model(self, model_name: str) -> "SentimentAnalyzerBuilder":
        self._model_name = model_name
        return self

    def build(self) -> "SentimentAnalyzer":
        return SentimentAnalyzer(
            use_lowercase=self._use_lowercase,
            threshold=self._threshold,
            model_name=self._model_name,
        )


class SentimentAnalyzer:
    def __init__(
        self,
        use_lowercase: bool = False,
        threshold: float = 0.5,
        model_name: Optional[str] = None,
    ) -> None:
        print(f"[SentimentAnalyzer] Loading model '{model_name}'...")
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentAnalyzer] Pipeline loaded.")
        self.use_lowercase: bool = use_lowercase
        self.threshold: float = threshold

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[SentimentAnalyzer] Original text: '{text}'")
        if self.use_lowercase:
            text = text.lower()
            print(f"[SentimentAnalyzer] Lowercased text: '{text}'")
        result: Dict[str, Any] = self.pipeline(text)[0]
        print(f"[SentimentAnalyzer] Raw model output: {result}")
        # Apply a custom threshold, e.g., for negative label
        if result["score"] < self.threshold:
            print(
                f"[SentimentAnalyzer] Score below threshold ({self.threshold}). Interpreted as NEGATIVE."
            )
            result["label"] = "NEGATIVE"
        return result


if __name__ == "__main__":
    print("\n--- Building Analyzer without lower case and threshold = 0.9 ---")
    analyzer1: SentimentAnalyzer = (
        SentimentAnalyzerBuilder().with_lowercase(False).with_threshold(0.9).build()
    )
    result1: Dict[str, Any] = analyzer1.analyze("I think I may like this product...")
    print(f"Result 1: {result1}")

    print("\n--- Building Analyzer with lowercasing and threshold = 0.99999 ---")
    analyzer2: SentimentAnalyzer = (
        SentimentAnalyzerBuilder().with_lowercase(True).with_threshold(0.99999).build()
    )
    result2: Dict[str, Any] = analyzer2.analyze("I think I may like this product...")
    print(f"Result 2: {result2}")
