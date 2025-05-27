"""
Prototype Pattern in Machine Learning/AI Model Configuration

The Prototype design pattern is used to create new objects by copying (cloning) an existing object (the prototype),
rather than building them from scratch. This is particularly useful in ML/AI contexts when object creation is expensive,
for example, when you have a pre-configured or partially trained model, and you want to rapidly generate similar instances
with slight modifications (e.g., different parameters or runtime options).

In this example, we use the Prototype pattern to efficiently clone and configure sentiment analyzers.
The key point is that all clones share the same Hugging Face pipeline object, which means:
    - Only one instance of the model is ever loaded into memory, regardless of how many analyzers you clone.
    - Cloning is extremely fast and memory efficient, since no additional models are loaded.
    - Each clone can have its own configuration (like preprocessing or thresholds), but inference is always performed
      using the exact same loaded model instance.
This is ideal when you want multiple "handlers" or "wrappers" with different behaviors, but don't want to reload
or duplicate the resource-intensive model.
"""

import copy
from typing import Any, Dict, Optional

from transformers import pipeline
from transformers.pipelines.base import Pipeline


class SentimentAnalyzerPrototype:
    def __init__(
        self,
        use_lowercase: bool = False,
        threshold: float = 0.5,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        pipeline_instance: Optional[Pipeline] = None,
    ) -> None:
        self.use_lowercase: bool = use_lowercase
        self.threshold: float = threshold
        self.model_name: str = model_name
        if pipeline_instance is not None:
            self.pipeline: Pipeline = pipeline_instance
        else:
            print(f"[SentimentAnalyzerPrototype] Loading model '{model_name}'...")
            self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
            print("[SentimentAnalyzerPrototype] Pipeline loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[SentimentAnalyzerPrototype] Original text: '{text}'")
        if self.use_lowercase:
            text = text.lower()
            print(f"[SentimentAnalyzerPrototype] Lowercased text: '{text}'")
        result: Dict[str, Any] = self.pipeline(text)[0]
        print(f"[SentimentAnalyzerPrototype] Raw model output: {result}")
        if result["score"] < self.threshold:
            print(
                f"[SentimentAnalyzerPrototype] Score below threshold ({self.threshold}). Interpreted as NEGATIVE."
            )
            result["label"] = "NEGATIVE"
        return result

    def clone(
        self, use_lowercase: Optional[bool] = None, threshold: Optional[float] = None
    ) -> "SentimentAnalyzerPrototype":
        """
        Clone the current instance, allowing for optional overrides of attributes.
        Reuses the same heavy pipeline to avoid redundant loading.
        """
        print("[SentimentAnalyzerPrototype] Cloning analyzer...")
        # Shallow copy is fine, but ensure the pipeline is shared, not duplicated
        clone_obj = copy.copy(self)
        if use_lowercase is not None:
            clone_obj.use_lowercase = use_lowercase
        if threshold is not None:
            clone_obj.threshold = threshold
        print(
            f"[SentimentAnalyzerPrototype] Clone created with use_lowercase={clone_obj.use_lowercase}, threshold={clone_obj.threshold}"
        )
        return clone_obj


if __name__ == "__main__":
    print("\n--- Creating original (prototype) analyzer ---")
    prototype: SentimentAnalyzerPrototype = SentimentAnalyzerPrototype(
        use_lowercase=False, threshold=0.5
    )
    result1: Dict[str, Any] = prototype.analyze("Maybe I like this product.")
    print(f"Prototype Result: {result1}")

    print("\n--- Cloning analyzer with lowercasing and a higher threshold ---")
    analyzer_clone1: SentimentAnalyzerPrototype = prototype.clone(
        use_lowercase=True, threshold=0.9
    )
    result2: Dict[str, Any] = analyzer_clone1.analyze("Maybe I like this product.")
    print(f"Clone 1 Result: {result2}")

    print("\n--- Cloning analyzer with different threshold only ---")
    analyzer_clone2: SentimentAnalyzerPrototype = prototype.clone(threshold=0.99999)
    result3: Dict[str, Any] = analyzer_clone2.analyze("Maybe I like this product.")
    print(f"Clone 2 Result: {result3}")

    print(
        "\n--- Confirming all clones share the same pipeline object (no extra loading) ---"
    )
    print(f"prototype.pipeline id: {id(prototype.pipeline)}")
    print(f"analyzer_clone1.pipeline id: {id(analyzer_clone1.pipeline)}")
    print(f"analyzer_clone2.pipeline id: {id(analyzer_clone2.pipeline)}")
    print(
        f"All share same pipeline: {prototype.pipeline is analyzer_clone1.pipeline is analyzer_clone2.pipeline}"
    )
