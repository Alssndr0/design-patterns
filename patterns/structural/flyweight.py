"""
Flyweight Pattern in Machine Learning/AI Model Serving

The Flyweight design pattern is used to minimize memory usage by sharing as much data as possible with similar objects.
In ML/AI, this is especially useful when you need to create many lightweight objects that share heavy resources,
such as model weights, embeddings, or tokenizers.

In this example:
- We need to analyze sentiment for many different users, each with their own preferences (e.g., threshold),
  but all can share the same Hugging Face model pipeline.
- The `SentimentAnalyzerFactory` keeps a cache of loaded models and returns shared instances (the flyweights).
- User-specific analyzers store only their preferences, but all use the same shared model to save memory and initialization time.
"""

from typing import Any, Dict

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- The Flyweight: Shared sentiment analysis model ---
class SentimentModelFlyweight:
    def __init__(self, model_name: str) -> None:
        print(f"[SentimentModelFlyweight] Loading model '{model_name}'...")
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentModelFlyweight] Model loaded.")

    def analyze(self, text: str) -> Dict[str, Any]:
        return self.pipeline(text)[0]


# --- The Flyweight Factory: Manages and reuses shared models ---
class SentimentAnalyzerFactory:
    _models: Dict[str, SentimentModelFlyweight] = {}

    @classmethod
    def get_model(cls, model_name: str) -> SentimentModelFlyweight:
        if model_name not in cls._models:
            print(
                f"[SentimentAnalyzerFactory] No cached model for '{model_name}', creating new flyweight..."
            )
            cls._models[model_name] = SentimentModelFlyweight(model_name)
        else:
            print(
                f"[SentimentAnalyzerFactory] Returning cached flyweight for '{model_name}'"
            )
        return cls._models[model_name]


# --- The lightweight context: User-specific analyzer ---
class UserSentimentAnalyzer:
    def __init__(
        self,
        user_id: str,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        threshold: float = 0.5,
    ) -> None:
        self.user_id = user_id
        self.threshold = threshold
        self.model_name = model_name
        # Get the shared flyweight (model)
        self.model = SentimentAnalyzerFactory.get_model(model_name)

    def analyze(self, text: str) -> Dict[str, Any]:
        print(f"[UserSentimentAnalyzer] User '{self.user_id}' analyzing: '{text}'")
        result = self.model.analyze(text)
        # User-specific postprocessing (threshold)
        if result["score"] < self.threshold:
            result = result.copy()
            result["label"] = "NEGATIVE"
        print(f"[UserSentimentAnalyzer] User '{self.user_id}' result: {result}")
        return result


if __name__ == "__main__":
    print("\n--- Creating analyzers for different users (all share model) ---")
    user1 = UserSentimentAnalyzer(user_id="alice", threshold=0.7)
    user2 = UserSentimentAnalyzer(user_id="bob", threshold=0.9)
    user3 = UserSentimentAnalyzer(user_id="carol")  # Default threshold

    # All analyzers use the same underlying model instance
    print("\n--- All users analyze text ---")
    for user in [user1, user2, user3]:
        user.analyze("I think this product is great!")

    print("\n--- Confirm model is shared (flyweight) ---")
    print(f"user1.model is user2.model: {user1.model is user2.model}")
    print(f"user1.model is user3.model: {user1.model is user3.model}")
