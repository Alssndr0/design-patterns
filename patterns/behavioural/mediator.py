"""
Mediator Pattern in Machine Learning/AI Component Orchestration

The Mediator design pattern defines an object (the mediator) that encapsulates how a set of objects interact.
This promotes loose coupling by preventing objects from referring to each other explicitly,
allowing their interaction to be varied independently.

In ML/AI, this is useful for orchestrating workflows where multiple components (preprocessors, models,
postprocessors, data stores) need to interact, but you want to avoid tangled dependencies and direct references.

In this example:
- Several components (preprocessor, sentiment analyzer, logger) interact only via the `Mediator`.
- The mediator controls the workflow, routing messages and data between components.
- This allows you to add or modify components without changing the logic in the other classes.
"""

from typing import Any, Dict, Optional

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Colleagues ---
class Preprocessor:
    def process(self, text: str) -> str:
        cleaned = text.lower().strip()
        print(f"[Preprocessor] Cleaned text: '{cleaned}'")
        return cleaned


class SentimentAnalyzer:
    def __init__(
        self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> None:
        print(f"[SentimentAnalyzer] Loading model '{model_name}'...")
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentAnalyzer] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[SentimentAnalyzer] Analyzing: '{text}'")
        return self.pipeline(text)[0]


class Logger:
    def log(self, message: str) -> None:
        print(f"[Logger] {message}")


# --- Mediator ---
class SentimentPipelineMediator:
    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self.preprocessor = preprocessor or Preprocessor()
        self.analyzer = analyzer or SentimentAnalyzer()
        self.logger = logger or Logger()

    def run_pipeline(self, text: str) -> Dict[str, Any]:
        self.logger.log("Pipeline started.")
        cleaned = self.preprocessor.process(text)
        self.logger.log(f"Text after preprocessing: '{cleaned}'")
        result = self.analyzer.analyze(cleaned)
        self.logger.log(f"Raw model result: {result}")
        self.logger.log("Pipeline complete.")
        return result


if __name__ == "__main__":
    print("\n--- Using Mediator for pipeline orchestration ---")
    mediator = SentimentPipelineMediator()
    input_text = "  I Absolutely LOVE this product!  "
    result = mediator.run_pipeline(input_text)
    print(f"Final Mediator Result: {result}")
