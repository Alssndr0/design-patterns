"""
Composite Pattern in Machine Learning/AI Model Pipelines

The Composite design pattern allows you to compose objects into tree structures to represent part-whole hierarchies.
It lets clients treat individual objects and compositions of objects uniformly.

In ML/AI, this is useful when you want to chain, stack, or aggregate several models or preprocessing steps,
treating a sequence of models or operations in the same way as a single model.

In this example:
- We define a `SentimentComponent` interface with a uniform `.analyze(text)` method.
- Both `SentimentAnalyzerLeaf` (a single model) and `SentimentComposite` (a sequence of components)
  implement this interface.
- The composite can contain leaves or other composites, enabling flexible and hierarchical workflows,
  such as applying multiple sentiment models and aggregating their results.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Component Interface ---
class SentimentComponent(ABC):
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        pass


# --- Leaf: Individual sentiment analyzer ---
class SentimentAnalyzerLeaf(SentimentComponent):
    def __init__(self, model_name: str) -> None:
        print(f"[SentimentAnalyzerLeaf] Loading model '{model_name}'...")
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentAnalyzerLeaf] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[SentimentAnalyzerLeaf] Analyzing: '{text}'")
        result = self.pipeline(text)[0]
        print(f"[SentimentAnalyzerLeaf] Result: {result}")
        return result


# --- Composite: Group of sentiment analyzers (can be nested) ---
class SentimentComposite(SentimentComponent):
    def __init__(self) -> None:
        self.children: List[SentimentComponent] = []

    def add(self, component: SentimentComponent) -> None:
        self.children.append(component)

    def remove(self, component: SentimentComponent) -> None:
        self.children.remove(component)

    def analyze(self, text: str) -> Dict[str, Any]:
        print("[SentimentComposite] Aggregating results from children...")
        results = [child.analyze(text) for child in self.children]
        # Simple aggregation: majority voting or average score
        positive = sum(1 for r in results if r["label"].upper().startswith("POS"))
        negative = sum(1 for r in results if r["label"].upper().startswith("NEG"))
        agg_label = "POSITIVE" if positive >= negative else "NEGATIVE"
        agg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
        agg_result = {"label": agg_label, "score": agg_score, "details": results}
        print(f"[SentimentComposite] Aggregated result: {agg_result}")
        return agg_result


if __name__ == "__main__":
    print("\n--- Single model (Leaf) ---")
    leaf1 = SentimentAnalyzerLeaf("distilbert-base-uncased-finetuned-sst-2-english")
    result1 = leaf1.analyze("I think this product is great!")
    print(f"Leaf Result: {result1}")

    print("\n--- Composite of two different models ---")
    composite = SentimentComposite()
    composite.add(leaf1)
    # Add a second model (simulate diversity)
    leaf2 = SentimentAnalyzerLeaf("nlptown/bert-base-multilingual-uncased-sentiment")
    composite.add(leaf2)
    result2 = composite.analyze("I think this product is great!")
    print(f"Composite Result: {result2}")

    print("\n--- Nested composite (Composite of composites) ---")
    super_composite = SentimentComposite()
    super_composite.add(composite)
    # Add another model directly to the super composite
    leaf3 = SentimentAnalyzerLeaf("finiteautomata/bertweet-base-sentiment-analysis")
    super_composite.add(leaf3)
    result3 = super_composite.analyze("I think this product is great!")
    print(f"Super Composite Result: {result3}")
