"""
State Pattern in Machine Learning/AI Pipeline Modes

The State design pattern allows an object to alter its behavior when its internal state changes.
The object will appear to change its class by delegating requests to state-specific objects.

In ML/AI, this is useful for:
- Switching pipeline or model behavior based on operational mode (idle, training, serving, error)
- Managing complex workflows with distinct processing phases
- Handling "mode switching" in serving APIs or data pipelines

In this example:
- `SentimentPipelineContext` holds a reference to a `State` object and delegates calls to it.
- Different states (`IdleState`, `ServingState`, `ErrorState`) control how inference requests are handled.
- The context can switch states dynamically, and its behavior changes accordingly.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


# --- State interface ---
class PipelineState(ABC):
    @abstractmethod
    def handle(self, context: "SentimentPipelineContext", text: str) -> Dict[str, Any]:
        pass


# --- Concrete States ---
class IdleState(PipelineState):
    def handle(self, context: "SentimentPipelineContext", text: str) -> Dict[str, Any]:
        print("[IdleState] Pipeline is idle. Switching to serving state...")
        context.set_state(ServingState())
        return context.handle_request(text)


class ServingState(PipelineState):
    def handle(self, context: "SentimentPipelineContext", text: str) -> Dict[str, Any]:
        print("[ServingState] Running sentiment analysis...")
        # Simulate sentiment analysis (real model could be called here)
        result = {
            "label": "POSITIVE" if "good" in text.lower() else "NEGATIVE",
            "score": 0.88,
        }
        print(f"[ServingState] Result: {result}")
        # Example: Simulate error if certain text is seen
        if "error" in text.lower():
            print("[ServingState] Error encountered. Switching to error state.")
            context.set_state(ErrorState())
        return result


class ErrorState(PipelineState):
    def handle(self, context: "SentimentPipelineContext", text: str) -> Dict[str, Any]:
        print("[ErrorState] Cannot process request due to pipeline error. Resetting...")
        # After error, reset to idle
        context.set_state(IdleState())
        return {"error": "Pipeline in error state. Resetting to idle."}


# --- Context: Maintains state reference and delegates behavior ---
class SentimentPipelineContext:
    def __init__(self) -> None:
        self._state: PipelineState = IdleState()

    def set_state(self, state: PipelineState) -> None:
        print(
            f"[SentimentPipelineContext] State changed to: {state.__class__.__name__}"
        )
        self._state = state

    def handle_request(self, text: str) -> Dict[str, Any]:
        return self._state.handle(self, text)


if __name__ == "__main__":
    print("\n--- State Pattern: Pipeline with Modes ---")
    pipeline = SentimentPipelineContext()

    print("\nFirst request (pipeline is idle):")
    print(pipeline.handle_request("This is a good product!"))

    print("\nSecond request (now serving):")
    print(pipeline.handle_request("This is a bad experience."))

    print("\nThird request (simulate error):")
    print(pipeline.handle_request("Trigger an error here."))

    print("\nFourth request (should be idle again):")
    print(pipeline.handle_request("Another review."))
