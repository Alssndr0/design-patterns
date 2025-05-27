"""
Iterator Pattern in Machine Learning/AI Data Processing

The Iterator design pattern provides a way to access elements of a collection sequentially,
without exposing its underlying representation. In ML/AI, this is useful for
processing batches of data, streaming records, or chaining multiple data sources
in a consistent and reusable way.

In this example:
- We define a `Dataset` that holds a list of text samples.
- The `DatasetIterator` provides a uniform way to iterate over data for inference.
- The client code uses the iterator to process each sample with a model, without knowing
  anything about the collection's implementation.

This pattern is essential for batching, streaming inference, and pipeline construction in ML workflows.
"""

from typing import Any, Dict, Iterator, List

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- The Aggregate: Collection of text samples ---
class Dataset:
    def __init__(self, samples: List[str]) -> None:
        self.samples = samples

    def __iter__(self) -> "DatasetIterator":
        return DatasetIterator(self)


# --- The Iterator ---
class DatasetIterator(Iterator[str]):
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._index = 0

    def __next__(self) -> str:
        if self._index < len(self._dataset.samples):
            result = self._dataset.samples[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration()

    def __iter__(self) -> "DatasetIterator":
        return self


# --- Example: Using the iterator in ML inference ---
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


if __name__ == "__main__":
    # Example data
    samples = [
        "I love this product!",
        "Not bad, but not great either.",
        "Absolutely terrible experience.",
        "It's okay, could be better.",
    ]
    dataset = Dataset(samples)
    analyzer = SentimentAnalyzer()

    print("\n--- Iterating over dataset for sentiment analysis ---")
    for text in dataset:
        result = analyzer.analyze(text)
        print(f"Text: '{text}' => Sentiment: {result}")
