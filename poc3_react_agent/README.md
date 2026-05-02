# PoC 3 — ReAct Agent with Tools

A minimal command-line **ReAct** (Reason + Act) agent in Python. The agent
gets a natural-language goal, decides on its own whether to look something
up or do a calculation, and stops when it can answer — without us
hard-coding the control flow.

Reference: Yao et al., *ReAct: Synergizing Reasoning and Acting in Language
Models*, ICLR 2023.

## What this PoC shows

| Concept                     | Where to look                                       |
| --------------------------- | --------------------------------------------------- |
| LLM-as-controller           | `agent.py` → `run_agent()`                          |
| Tool calling (text-format)  | `SYSTEM_PROMPT` + `_ACTION_RE` parser               |
| Three tools (search / calculator / final_answer) | `tools.py` → `TOOLS` dict      |
| Safe arithmetic evaluation  | `tools.py` → `_safe_eval()` (no `eval()`!)          |
| Hard step budget guardrail  | `MAX_STEPS = 6`                                     |
| Auditable trace             | every Thought / Action / Observation is printed     |

## The ReAct loop

```
       ┌──────────────────────────────────────────┐
       │                                          ▼
   ┌─────────┐    Action     ┌─────────┐   Observation
   │ Thought │ ────────────► │  Tool   │ ───────────────┐
   └─────────┘               └─────────┘                │
       ▲                                                │
       └────────────────────────────────────────────────┘
                   (until final_answer is called)
```

The LLM emits a strict text block:

```
Thought: I need to look up the capital of France.
Action: search
Action Input: capital of france
```

The runtime parses it with a regex, runs `TOOLS["search"]("capital of france")`,
appends `Observation: Paris` back into the conversation, and the loop
repeats until the model calls `final_answer`.

## Setup

```bash
cd poc3_react_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then edit .env: OPENROUTER_API_KEY=sk-or-v1-...
```

The agent uses the OpenAI SDK pointed at OpenRouter
(`https://openrouter.ai/api/v1`). Default model: `anthropic/claude-sonnet-4`.
Change `LLM_MODEL` in `agent.py` to use any
[OpenRouter slug](https://openrouter.ai/models).

## Run

```bash
# Run with the two built-in example goals:
python agent.py

# Or pass your own goal:
python agent.py "What is the capital of Japan, and what is 8 squared?"
```

You'll see every step printed:

```
=== GOAL: What is the capital of Japan, and what is 8 squared? ===

[step 1] Thought: I need two pieces of information...
[step 1] Action: search
[step 1] Action Input: capital of japan
[step 1] Observation: Tokyo

[step 2] Thought: Now I'll compute 8 squared.
[step 2] Action: calculator
[step 2] Action Input: 8 ** 2
[step 2] Observation: 64

[step 3] Thought: I have both. Returning the final answer.
[step 3] Action: final_answer
[step 3] Action Input: The capital of Japan is Tokyo, and 8 squared is 64.
[step 3] Observation: The capital of Japan is Tokyo, and 8 squared is 64.

=== FINAL ANSWER: ... ===
```

## Files

- [agent.py](agent.py) — the ReAct loop, system prompt, regex parser, OpenRouter client.
- [tools.py](tools.py) — `calculator`, `search`, `final_answer` + the `TOOLS` dict.
- [requirements.txt](requirements.txt)
- [.env.example](.env.example) — copy to `.env` and add your OpenRouter key.
- [.gitignore](.gitignore)

## Guardrails worth noting

- **`max_steps=6`** caps the loop — the single most important guardrail for
  any agent that bills per token.
- **`stop=["Observation:"]`** in the LLM call prevents the model from
  hallucinating its own observations.
- **`_safe_eval`** parses the calculator input as an AST and only allows
  arithmetic nodes — no `eval()`, no imports, no attribute access.
- **`final_answer` is a tool** — the loop's stopping condition is just
  "the LLM called final_answer", which is easy to reason about.

## Ideas for the next iteration

- Add a `rag_search` tool that calls the RAG app from
  [poc1_rag_pdf](../poc1_rag_pdf/) so the agent can answer questions about
  uploaded PDFs.
- Add a `web_search` tool (e.g. via DuckDuckGo) for live information.
- Persist a long-term memory in a JSON file to remember facts across runs.
- Swap the regex protocol for native [OpenAI-style tool calling](https://platform.openai.com/docs/guides/function-calling)
  — more robust at the cost of being model-specific.
