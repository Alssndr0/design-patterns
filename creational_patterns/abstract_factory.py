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

from PIL import Image
from transformers import pipeline


# --- Abstract Products ---
class ModelLoader(ABC):
    @abstractmethod
    def load(self):
        pass


class Analyzer(ABC):
    @abstractmethod
    def analyze(self, data):
        pass


# --- Concrete Products for NLP (Sentiment) ---
class SentimentModelLoader(ModelLoader):
    def load(self):
        print("[SentimentModelLoader] Loading sentiment analysis model...")
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )


class SentimentAnalyzer(Analyzer):
    def __init__(self, model):
        self.model = model

    def analyze(self, text):
        print(f"[SentimentAnalyzer] Analyzing text: '{text}'")
        return self.model(text)


# --- Concrete Products for Vision (Image Classification) ---
class ImageModelLoader(ModelLoader):
    def load(self):
        print("[ImageModelLoader] Loading image classification model...")
        # This pipeline expects a PIL image input (for demo, model is small and quick)
        return pipeline("image-classification", model="google/vit-base-patch16-224")


class ImageAnalyzer(Analyzer):
    def __init__(self, model):
        self.model = model

    def analyze(self, image):
        print("[ImageAnalyzer] Analyzing image...")
        return self.model(image)


# --- Abstract Factory ---
class AIPipelineFactory(ABC):
    @abstractmethod
    def create_loader(self):
        pass

    @abstractmethod
    def create_analyzer(self, model):
        pass


# --- Concrete Factories ---
class NLPPipelineFactory(AIPipelineFactory):
    def create_loader(self):
        return SentimentModelLoader()

    def create_analyzer(self, model):
        return SentimentAnalyzer(model)


class VisionPipelineFactory(AIPipelineFactory):
    def create_loader(self):
        return ImageModelLoader()

    def create_analyzer(self, model):
        return ImageAnalyzer(model)


# --- Client code that uses the abstract factory ---
def run_pipeline(factory, data):
    loader = factory.create_loader()
    model = loader.load()
    analyzer = factory.create_analyzer(model)
    result = analyzer.analyze(data)
    print(f"Result: {result}")


if __name__ == "__main__":
    print("\n--- Using NLP pipeline (Sentiment Analysis) ---")
    nlp_factory = NLPPipelineFactory()
    run_pipeline(nlp_factory, "This is a fantastic product!")

    print("\n--- Using Vision pipeline (Image Classification) ---")
    vision_factory = VisionPipelineFactory()

    img_url = "../assets/puppy.jpg"
    image = Image.open(img_url)
    run_pipeline(vision_factory, image)
