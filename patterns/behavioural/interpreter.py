"""
Interpreter Pattern in Machine Learning/AI Mini-Language Execution

The Interpreter design pattern defines a representation for a language’s grammar and an interpreter
that uses that representation to interpret sentences in the language.
In ML/AI, you can use it to parse and execute simple domain-specific commands for building pipelines,
applying preprocessing, or expressing logic in a flexible, extensible way.

In this example:
- We define a tiny expression language for text processing pipelines (e.g., "LOWERCASE", "REMOVE_PUNCT", "SENTIMENT").
- Each command in the pipeline is an expression; the interpreter executes the sequence.
- New commands/expressions can be added easily, enabling non-programmers or configs to describe workflows.

This enables flexible, data-driven, and extensible pipeline configuration and execution.
"""

import string
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from transformers import pipeline
from transformers.pipelines.base import Pipeline


# --- Abstract Expression ---
class Expression(ABC):
    @abstractmethod
    def interpret(self, context: Dict[str, Any]) -> None:
        pass


# --- Concrete Expressions ---
class LowercaseExpression(Expression):
    def interpret(self, context: Dict[str, Any]) -> None:
        text = context["text"]
        lowered = text.lower()
        print(f"[LowercaseExpression] '{text}' -> '{lowered}'")
        context["text"] = lowered


class RemovePunctuationExpression(Expression):
    def interpret(self, context: Dict[str, Any]) -> None:
        text = context["text"]
        no_punct = text.translate(str.maketrans("", "", string.punctuation))
        print(f"[RemovePunctuationExpression] '{text}' -> '{no_punct}'")
        context["text"] = no_punct


class SentimentExpression(Expression):
    def __init__(
        self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> None:
        self.pipeline: Pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[SentimentExpression] Model loaded.")

    def interpret(self, context: Dict[str, Any]) -> None:
        text = context["text"]
        print(f"[SentimentExpression] Analyzing: '{text}'")
        result = self.pipeline(text)[0]
        context["sentiment"] = result
        print(f"[SentimentExpression] Result: {result}")


# --- The Interpreter: runs a sequence of expressions ---
class PipelineInterpreter:
    def __init__(self, expressions: List[Expression]) -> None:
        self.expressions = expressions

    def interpret(self, text: str) -> Dict[str, Any]:
        context: Dict[str, Any] = {"text": text}
        for expr in self.expressions:
            expr.interpret(context)
        return context


if __name__ == "__main__":
    print("\n--- Interpreter Pattern: Custom Pipeline ---")
    # Define a "program" as a sequence of commands
    program: List[Expression] = [
        LowercaseExpression(),
        RemovePunctuationExpression(),
        SentimentExpression(),
    ]

    interpreter = PipelineInterpreter(program)

    text = "This PRODUCT is, without a doubt, AMAZING!!!"
    final_context = interpreter.interpret(text)
    print("\nFinal Interpreter Context:", final_context)
    print("Sentiment Result:", final_context.get("sentiment"))
