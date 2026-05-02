"""A minimal ReAct (Reason + Act) agent.

The LLM emits a strict
    Thought: ...
    Action: <tool_name>
    Action Input: <text>
block. The runtime parses it with regex, executes the named tool from
`tools.TOOLS`, appends the Observation back into the conversation, and loops
until `final_answer` is called or the step budget is exhausted.

We use OpenRouter's OpenAI-compatible endpoint so any model slug works.
"""

from __future__ import annotations

import os
import re
import sys
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_DESCRIPTIONS, TOOLS

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "anthropic/claude-sonnet-4"
MAX_STEPS = 6

SYSTEM_PROMPT = f"""You are a helpful ReAct agent. You solve the user's goal
step by step using a small toolbox. After every Thought you MUST emit exactly
one Action and one Action Input, in the following format and nothing else:

Thought: <your reasoning>
Action: <one of: {", ".join(TOOLS)}>
Action Input: <the argument string>

The runtime will execute the action and reply with:

Observation: <result>

Then you continue with the next Thought / Action / Action Input.
When you have the final answer, call:

Action: final_answer
Action Input: <the answer to give the user>

Available tools:
""" + "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()) + """

Rules:
- Output ONLY the Thought/Action/Action Input block; do not add anything after it.
- Do not invent tool names; use only those listed above.
- Use `final_answer` as soon as you can answer the question.
"""


_ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*"
    r"Action:\s*(?P<action>[a-zA-Z_]+)\s*"
    r"Action Input:\s*(?P<input>.*?)\s*$",
    re.DOTALL,
)


def _parse(response: str):
    m = _ACTION_RE.search(response.strip())
    if not m:
        return None
    return m.group("thought").strip(), m.group("action").strip(), m.group("input").strip()


def _client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env).")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def run_agent(goal: str, max_steps: int = MAX_STEPS) -> str:
    client = _client()
    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Goal: {goal}"},
    ]

    print(f"\n=== GOAL: {goal} ===")
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.0,
            stop=["Observation:"],  # don't let the model hallucinate observations
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _parse(raw)
        if not parsed:
            print(f"[step {step}] Could not parse model output:\n{raw}")
            return "ERROR: malformed agent output."

        thought, action, action_input = parsed
        print(f"\n[step {step}] Thought: {thought}")
        print(f"[step {step}] Action: {action}")
        print(f"[step {step}] Action Input: {action_input}")

        if action not in TOOLS:
            observation = f"ERROR: unknown tool '{action}'. Pick one of: {', '.join(TOOLS)}."
        else:
            try:
                observation = TOOLS[action](action_input)
            except Exception as e:  # noqa: BLE001
                observation = f"ERROR: {e}"

        print(f"[step {step}] Observation: {observation}")

        # Append the assistant turn (its Thought/Action block) and then the
        # observation as a user message — this is the ReAct convention.
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        if action == "final_answer":
            print(f"\n=== FINAL ANSWER: {observation} ===")
            return observation

    print(f"\n=== STOPPED: max_steps={max_steps} reached ===")
    return "ERROR: step budget exhausted."


if __name__ == "__main__":
    sample_goals = [
        "What is the capital of France, and how many letters does it have times 2?",
        "Who wrote 1984, and what is 17 * 23?",
    ]
    if len(sys.argv) > 1:
        sample_goals = [" ".join(sys.argv[1:])]

    for g in sample_goals:
        run_agent(g)
        print("\n" + "=" * 70)
