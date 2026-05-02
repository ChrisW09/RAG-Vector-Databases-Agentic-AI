"""Minimal toolbox for the ReAct agent.

Each tool is a plain Python callable that takes a single string argument and
returns a string observation. The TOOLS dict is the contract between the LLM
(which emits an action name + input) and the runtime (which executes it).
"""

from __future__ import annotations

import ast
import operator as op
from typing import Callable, Dict

# --- Safe arithmetic evaluator ------------------------------------------------
# We deliberately avoid eval(): it would let the LLM execute arbitrary code.
_ALLOWED_BINOPS: Dict[type, Callable] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARYOPS: Dict[type, Callable] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Disallowed expression: {ast.dump(node)}")


def calculator(expr: str) -> str:
    """Evaluate a pure arithmetic expression like '2 * (3 + 4)'."""
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


# --- Toy in-memory knowledge base --------------------------------------------
_KB: Dict[str, str] = {
    "capital of france": "Paris",
    "capital of germany": "Berlin",
    "capital of japan": "Tokyo",
    "capital of brazil": "Brasilia",
    "capital of australia": "Canberra",
    "speed of light": "299792458 m/s",
    "pi": "3.14159265",
    "founder of microsoft": "Bill Gates and Paul Allen",
    "author of 1984": "George Orwell",
}


def search(query: str) -> str:
    """Lookup a fact in the toy KB. Case-insensitive substring match."""
    q = query.lower().strip().rstrip("?.! ")
    if q in _KB:
        return _KB[q]
    for key, value in _KB.items():
        if key in q or q in key:
            return value
    return f"NOT FOUND: '{query}'. Known keys: {', '.join(sorted(_KB))}"


def final_answer(text: str) -> str:
    """Terminal action: returns the final answer back to the user."""
    return text


TOOLS: Dict[str, Callable[[str], str]] = {
    "calculator": calculator,
    "search": search,
    "final_answer": final_answer,
}

TOOL_DESCRIPTIONS = {
    "calculator": "Evaluate an arithmetic expression. Input: a math expression, e.g. '7 * 6'.",
    "search": "Look up a fact in a small knowledge base. Input: a short query, e.g. 'capital of france'.",
    "final_answer": "Return the final answer to the user. Input: the answer text. Use this to STOP.",
}
