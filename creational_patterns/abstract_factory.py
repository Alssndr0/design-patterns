"""
Abstract Factory Pattern in Machine Learning/AI Model Pipelines

The Abstract Factory pattern provides an interface for creating families of related or dependent objects
without specifying their concrete classes. In other words, it lets you create sets of objects that are
meant to work together, without coupling your code to their specific implementations.

In ML/AI engineering, this is useful when you want to offer configurable, pluggable model pipelines.
For instance, you might want to easily swap between NLP and Vision pipelines, or between different
providers (e.g., Hugging Face vs. custom models), without changing client code.

In this example, we use the Abstract Factory pattern to provide two different types of AI pipelines:
one for text sentiment analysis (NLP) and one for image classification (Vision). Each pipeline family
includes both a loader (for the model) and an analyzer (for inference), and you can create either
family with the same interface.
"""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image
from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Abstract Products ---
class ModelLoader(ABC):
    @abstractmethod
    def load(self, model: str) -> Pipeline:
        pass


class Analyzer(ABC):
    @abstractmethod
    def analyze(self, data: Any) -> Any:
        pass


# --- Concrete Products for NLP (Sentiment) ---
class SentimentModelLoader(ModelLoader):
    def load(self, model: str) -> Pipeline:
        print("[SentimentModelLoader] Loading sentiment analysis model...")
        return pipeline(
            "sentiment-analysis",
            model=model,
        )


class SentimentAnalyzer(Analyzer):
    def __init__(self, model: Pipeline) -> None:
        self.model: Pipeline = model

    def analyze(self, text: str) -> Any:
        print(f"[SentimentAnalyzer] Analyzing text: '{text}'")
        return self.model(text)


# --- Concrete Products for Vision (Image Classification) ---
class ImageModelLoader(ModelLoader):
    def load(self, model: str) -> Pipeline:
        print("[ImageModelLoader] Loading image classification model...")
        return pipeline("image-classification", model=model)


class ImageAnalyzer(Analyzer):
    def __init__(self, model: Pipeline) -> None:
        self.model: Pipeline = model

    def analyze(self, image: Image.Image) -> Any:
        print("[ImageAnalyzer] Analyzing image...")
        return self.model(image)


# --- Abstract Factory ---
class AIPipelineFactory(ABC):
    @abstractmethod
    def create_loader(self) -> ModelLoader:
        pass

    @abstractmethod
    def create_analyzer(self, model: Pipeline) -> Analyzer:
        pass


# --- Concrete Factories ---
class NLPPipelineFactory(AIPipelineFactory):
    def create_loader(self) -> ModelLoader:
        return SentimentModelLoader()

    def create_analyzer(self, model: Pipeline) -> Analyzer:
        return SentimentAnalyzer(model)


class VisionPipelineFactory(AIPipelineFactory):
    def create_loader(self) -> ModelLoader:
        return ImageModelLoader()

    def create_analyzer(self, model: Pipeline) -> Analyzer:
        return ImageAnalyzer(model)


# --- Client code that uses the abstract factory ---
def run_pipeline(factory: AIPipelineFactory, model: str, data: Any) -> None:
    loaded_model: Pipeline = factory.create_loader().load(model)
    analyzer: Analyzer = factory.create_analyzer(loaded_model)
    result: Any = analyzer.analyze(data)
    print(f"Result: {result}")


if __name__ == "__main__":
    print("\n--- Using NLP pipeline (Sentiment Analysis) ---")
    nlp_factory: AIPipelineFactory = NLPPipelineFactory()
    run_pipeline(
        nlp_factory,
        model="distilbert-base-uncased-finetuned-sst-2-english",
        data="This is a fantastic product!",
    )

    print("\n--- Using Vision pipeline (Image Classification) ---")
    img_path: str = "assets/puppy.jpg"
    image: Image.Image = Image.open(img_path)
    vision_factory: AIPipelineFactory = VisionPipelineFactory()
    run_pipeline(vision_factory, model="google/vit-base-patch16-224", data=image)
