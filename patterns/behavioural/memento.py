"""
Memento Pattern in Machine Learning/AI Model State Management

The Memento design pattern captures and restores an object's internal state without exposing its implementation details.
This is especially useful in ML/AI for saving and restoring model configurations, hyperparameters, or tuning progress,
enabling undo, checkpointing, or rollback to previous settings.

In this example:
- The `SentimentAnalyzer` can save and restore its configuration (like model name and threshold) using a `Memento`.
- The `Caretaker` manages multiple saved states, enabling undo/rollback for the client.
- This is handy for experimentation, hyperparameter tuning, or interactive pipelines.

No model weights are persisted here (use checkpoints for that), but configurations and parameters are safely snapshotted and restored.
"""

from typing import Any, Dict, List


# --- The Memento: stores state ---
class Memento:
    def __init__(self, state: Dict[str, Any]) -> None:
        self._state = state.copy()

    def get_state(self) -> Dict[str, Any]:
        return self._state.copy()


# --- The Originator: the object whose state we care about ---
class SentimentAnalyzer:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold

    def configure(self, model_name: str, threshold: float) -> None:
        print(
            f"[SentimentAnalyzer] Reconfiguring to model '{model_name}', threshold={threshold}"
        )
        self.model_name = model_name
        self.threshold = threshold

    def get_state(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "threshold": self.threshold}

    def set_state(self, state: Dict[str, Any]) -> None:
        print(f"[SentimentAnalyzer] Restoring state: {state}")
        self.model_name = state["model_name"]
        self.threshold = state["threshold"]

    def save(self) -> Memento:
        print("[SentimentAnalyzer] Saving current state.")
        return Memento(self.get_state())

    def restore(self, memento: Memento) -> None:
        print("[SentimentAnalyzer] Restoring from memento.")
        self.set_state(memento.get_state())


# --- The Caretaker: manages mementos (history) ---
class Caretaker:
    def __init__(self) -> None:
        self._history: List[Memento] = []

    def save(self, memento: Memento) -> None:
        self._history.append(memento)

    def undo(self) -> Memento:
        if not self._history:
            raise IndexError("No mementos to restore.")
        return self._history.pop()


if __name__ == "__main__":
    print("\n--- Memento Pattern: Save and Restore Configurations ---")
    caretaker = Caretaker()
    analyzer = SentimentAnalyzer()
    print(f"Initial State: {analyzer.get_state()}")

    # Save initial config
    caretaker.save(analyzer.save())

    # Change config for experiment 1
    analyzer.configure("nlptown/bert-base-multilingual-uncased-sentiment", 0.8)
    caretaker.save(analyzer.save())

    # Change config for experiment 2
    analyzer.configure("finiteautomata/bertweet-base-sentiment-analysis", 0.9)
    caretaker.save(analyzer.save())

    print(f"State before undo: {analyzer.get_state()}")
    print("\nUndoing last change...")
    analyzer.restore(caretaker.undo())
    print(f"Restored State: {analyzer.get_state()}")

    print("\nUndoing again...")
    analyzer.restore(caretaker.undo())
    print(f"Restored State: {analyzer.get_state()}")
