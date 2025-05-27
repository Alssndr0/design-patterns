"""
Proxy Pattern in Machine Learning/AI Model Serving

The Proxy design pattern provides a surrogate or placeholder for another object to control access to it.
This is useful in ML/AI when you want to:
- Add security, logging, caching, or lazy loading to expensive model calls,
- Hide the real model behind a simple interface,
- Defer or restrict access to a heavy or remote resource.

In this example:
- The `SentimentAnalyzer` class is the "real subject" that performs inference.
- The `SentimentAnalyzerProxy` controls access to the analyzer, adding logging, rate limiting, and lazy initialization.
- The client interacts only with the proxy, which manages when and how the real model is used.

You can use proxies for:
- Authorization/authentication,
- Remote model serving (gRPC/REST),
- Result caching,
- Monitoring or metering usage,
and more.
"""

from typing import Any, Dict, Optional

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Real Subject: The actual sentiment analyzer ---
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


# --- Proxy: Controls access to the real model ---
class SentimentAnalyzerProxy:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        max_calls: int = 3,
    ) -> None:
        self.model_name = model_name
        self.max_calls = max_calls
        self.calls_made = 0
        self._real_analyzer: Optional[SentimentAnalyzer] = None

    def _initialize_real_analyzer(self) -> None:
        if self._real_analyzer is None:
            print("[SentimentAnalyzerProxy] Initializing real SentimentAnalyzer...")
            self._real_analyzer = SentimentAnalyzer(self.model_name)

    def analyze(self, text: str) -> Dict[str, Any]:
        # Rate limiting
        if self.calls_made >= self.max_calls:
            print("[SentimentAnalyzerProxy] Rate limit exceeded!")
            return {"error": "Rate limit exceeded. Please try again later."}
        self._initialize_real_analyzer()
        self.calls_made += 1
        print(f"[SentimentAnalyzerProxy] Logging: call #{self.calls_made}")
        # Delegates to the real analyzer
        return self._real_analyzer.analyze(text)


if __name__ == "__main__":
    print("\n--- Using SentimentAnalyzerProxy ---")
    proxy = SentimentAnalyzerProxy(max_calls=2)

    texts = [
        "This is the best experience I've ever had!",
        "Not bad, but could be better.",
        "I really dislike this product.",
    ]
    for text in texts:
        result = proxy.analyze(text)
        print(f"Proxy Result: {result}")
