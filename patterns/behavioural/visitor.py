"""
Visitor Pattern in Machine Learning/AI Model and Pipeline Analytics

The Visitor design pattern lets you define new operations on a set of objects without changing their classes.
It decouples algorithms from the data structures on which they operate.

In ML/AI, this is useful for:
- Running analytics, reporting, or statistics across many pipeline components
- Collecting metadata or exporting configs from diverse model or step types
- Implementing inspection, validation, or transformation routines without altering component classes

In this example:
- Several pipeline components (`Preprocessor`, `SentimentModel`, `Postprocessor`) each accept a visitor.
- The `Visitor` interface defines operations for each component type.
- Concrete visitors perform analytics, export, or logging across all pipeline elements.

You can add new operations easily by just defining a new visitor.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


# --- Element (Component) interface ---
class PipelineComponent(ABC):
    @abstractmethod
    def accept(self, visitor: "Visitor") -> None:
        pass


# --- Concrete Elements ---
class Preprocessor(PipelineComponent):
    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def accept(self, visitor: "Visitor") -> None:
        visitor.visit_preprocessor(self)


class SentimentModel(PipelineComponent):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def accept(self, visitor: "Visitor") -> None:
        visitor.visit_sentiment_model(self)


class Postprocessor(PipelineComponent):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def accept(self, visitor: "Visitor") -> None:
        visitor.visit_postprocessor(self)


# --- Visitor interface ---
class Visitor(ABC):
    @abstractmethod
    def visit_preprocessor(self, pre: Preprocessor) -> None:
        pass

    @abstractmethod
    def visit_sentiment_model(self, model: SentimentModel) -> None:
        pass

    @abstractmethod
    def visit_postprocessor(self, post: Postprocessor) -> None:
        pass


# --- Concrete Visitor: Analytics ---
class AnalyticsVisitor(Visitor):
    def __init__(self) -> None:
        self.stats: Dict[str, Any] = {}

    def visit_preprocessor(self, pre: Preprocessor) -> None:
        print(f"[AnalyticsVisitor] Preprocessor: lowercase={pre.lowercase}")
        self.stats["pre_lowercase"] = pre.lowercase

    def visit_sentiment_model(self, model: SentimentModel) -> None:
        print(f"[AnalyticsVisitor] SentimentModel: model_name='{model.model_name}'")
        self.stats["model_name"] = model.model_name

    def visit_postprocessor(self, post: Postprocessor) -> None:
        print(f"[AnalyticsVisitor] Postprocessor: threshold={post.threshold}")
        self.stats["post_threshold"] = post.threshold


# --- Concrete Visitor: Export config ---
class ExportVisitor(Visitor):
    def __init__(self) -> None:
        self.export: Dict[str, Any] = {}

    def visit_preprocessor(self, pre: Preprocessor) -> None:
        self.export["preprocessor"] = {"lowercase": pre.lowercase}

    def visit_sentiment_model(self, model: SentimentModel) -> None:
        self.export["model"] = {"model_name": model.model_name}

    def visit_postprocessor(self, post: Postprocessor) -> None:
        self.export["postprocessor"] = {"threshold": post.threshold}


if __name__ == "__main__":
    # Compose pipeline components
    pipeline_components: List[PipelineComponent] = [
        Preprocessor(lowercase=True),
        SentimentModel(model_name="distilbert-base-uncased-finetuned-sst-2-english"),
        Postprocessor(threshold=0.75),
    ]

    print("\n--- Visitor Pattern: Analytics ---")
    analytics = AnalyticsVisitor()
    for component in pipeline_components:
        component.accept(analytics)
    print("Analytics stats:", analytics.stats)

    print("\n--- Visitor Pattern: Export config ---")
    exporter = ExportVisitor()
    for component in pipeline_components:
        component.accept(exporter)
    print("Exported config:", exporter.export)
