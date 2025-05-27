"""
Command Pattern in Machine Learning/AI Operations

The Command design pattern encapsulates a request as an object, allowing you to parameterize clients with different requests,
queue or log requests, and support undoable operations. In ML/AI, this is useful for decoupling operations such as inference,
model switching, data transformations, or batch predictions from their execution, making workflows flexible and extensible.

In this example:
- Each operation (e.g., run inference, switch model, batch predict) is represented as a Command object with a `.execute()` method.
- The client code can store, queue, and run commands in any order, without knowing the details of how they work.
- This makes it easy to build interactive tools, pipelines, or job queues in ML/AI projects.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Receiver: The object that knows how to perform the actions ---
class SentimentService:
    def __init__(
        self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> None:
        print(f"[SentimentService] Loading model '{model_name}'...")
        self.model_name = model_name
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentService] Model loaded.")

    def predict(self, text: str) -> Dict[str, Any]:
        print(f"[SentimentService] Predicting sentiment for: '{text}'")
        return self.pipeline(text)[0]

    def switch_model(self, model_name: str) -> None:
        print(f"[SentimentService] Switching to model '{model_name}'...")
        self.model_name = model_name
        self.pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentService] Model switched.")


# --- Command interface ---
class Command(ABC):
    @abstractmethod
    def execute(self) -> Any:
        pass


# --- Concrete Commands ---
class PredictCommand(Command):
    def __init__(self, service: SentimentService, text: str) -> None:
        self.service = service
        self.text = text

    def execute(self) -> Dict[str, Any]:
        return self.service.predict(self.text)


class SwitchModelCommand(Command):
    def __init__(self, service: SentimentService, new_model: str) -> None:
        self.service = service
        self.new_model = new_model

    def execute(self) -> None:
        self.service.switch_model(self.new_model)


class BatchPredictCommand(Command):
    def __init__(self, service: SentimentService, texts: List[str]) -> None:
        self.service = service
        self.texts = texts

    def execute(self) -> List[Dict[str, Any]]:
        print(f"[BatchPredictCommand] Predicting batch of {len(self.texts)} texts...")
        return [self.service.predict(text) for text in self.texts]


# --- Invoker: Stores and executes commands ---
class CommandInvoker:
    def __init__(self) -> None:
        self._commands: List[Command] = []

    def add_command(self, command: Command) -> None:
        self._commands.append(command)

    def run(self) -> None:
        for cmd in self._commands:
            result = cmd.execute()
            print(f"[CommandInvoker] Command result: {result}")


if __name__ == "__main__":
    # Set up service and commands
    service = SentimentService()
    invoker = CommandInvoker()

    invoker.add_command(PredictCommand(service, "I love this new feature!"))
    invoker.add_command(
        SwitchModelCommand(service, "nlptown/bert-base-multilingual-uncased-sentiment")
    )
    invoker.add_command(PredictCommand(service, "Me encanta este producto!"))
    invoker.add_command(
        BatchPredictCommand(
            service,
            ["Fantastic value for money.", "Not my favorite experience.", "Just okay."],
        )
    )

    print("\n--- Running all commands in order ---")
    invoker.run()
