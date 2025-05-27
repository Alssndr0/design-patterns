"""
Observer Pattern in Machine Learning/AI Event Notification

The Observer design pattern defines a one-to-many dependency between objects so that when one object (the subject) changes state,
all its dependents (observers) are notified and updated automatically.

In ML/AI, this is useful for:
- Logging or monitoring events (training, inference, etc.)
- Updating dashboards or UIs when model state or results change
- Triggering alerts or actions when results meet specific criteria

In this example:
- `SentimentAnalyzerSubject` notifies all registered observers when inference is performed.
- Observers include a logger, a result monitor, and a simple UI printer.
- Observers can be added/removed dynamically, and react to results in real time.
"""

from typing import Any, Dict, List, Protocol


# --- Observer interface ---
class Observer(Protocol):
    def update(self, result: Dict[str, Any]) -> None: ...


# --- Concrete Observers ---
class LoggerObserver:
    def update(self, result: Dict[str, Any]) -> None:
        print(f"[LoggerObserver] Logging result: {result}")


class AlertObserver:
    def update(self, result: Dict[str, Any]) -> None:
        if result.get("label", "") == "NEGATIVE":
            print("[AlertObserver] ALERT: Negative sentiment detected!")


class PrintObserver:
    def update(self, result: Dict[str, Any]) -> None:
        print(
            f"[PrintObserver] UI update: Sentiment = {result['label']}, Score = {result['score']:.3f}"
        )


# --- Subject: Notifies observers ---
class SentimentAnalyzerSubject:
    def __init__(self) -> None:
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self, result: Dict[str, Any]) -> None:
        for observer in self._observers:
            observer.update(result)

    def analyze(self, text: str) -> Dict[str, Any]:
        # For demo purposes, simulate result
        result = {
            "label": "NEGATIVE" if "bad" in text.lower() else "POSITIVE",
            "score": 0.82,
        }
        print(f"[SentimentAnalyzerSubject] Analyzed: '{text}' => {result}")
        self.notify(result)
        return result


if __name__ == "__main__":
    print("\n--- Observer Pattern: ML Event Notification ---")
    subject = SentimentAnalyzerSubject()

    logger = LoggerObserver()
    alert = AlertObserver()
    printer = PrintObserver()

    # Attach observers
    subject.attach(logger)
    subject.attach(alert)
    subject.attach(printer)

    # Run inference (all observers notified)
    print("\nFirst inference:")
    subject.analyze("This is a fantastic product!")

    print("\nSecond inference:")
    subject.analyze("This is a really bad experience.")

    # Detach one observer
    subject.detach(printer)
    print("\nThird inference (UI observer detached):")
    subject.analyze("Just okay.")
