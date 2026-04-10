---
name: add-chain
description: Step-by-step procedure to create a new LangChain LCEL chain in a genai-tk project and register it in the chain registry.
---

# Add an LCEL Chain

Follow these steps to add a new LangChain Expression Language (LCEL) chain to a genai-tk project.

## Prerequisites
- The project was initialized with `cli init`
- The project has a `<package>/chains/` directory

## Step 1: Create the Chain Module

Create a new file in `<package>/chains/my_chain.py`:

```python
"""My custom chain — describe what it does."""

from genai_tk.core.chain_registry import Example, RunnableItem, register_runnable
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.prompts import def_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable


def get_chain(config: dict | None = None) -> Runnable:
    """Create the chain: prompt | llm | parser."""
    llm = get_llm()
    prompt = def_prompt(
        system="You are an expert at [domain]. Be concise and accurate.",
        user="{input}",
    )
    return prompt | llm | StrOutputParser()


# Register so it appears in the webapp Runnable Playground
register_runnable(
    RunnableItem(
        tag="MyCategory",
        name="My Chain",
        runnable=get_chain,
        examples=[Example(query=["example input"])],
    )
)
```

## Step 2: Import the Chain (auto-registration)

The chain registers itself when imported. To ensure it's loaded, import it from a CLI command or add it to a module's `__init__.py`:

```python
# In <package>/chains/__init__.py
import <package_name>.chains.my_chain  # noqa: F401 — triggers registration
```

## Step 3: Test the Chain

### Via CLI (using the core run command)
```bash
uv run cli core run "My Chain" "example input"
```

### Via Webapp
Launch `make webapp` and navigate to the Runnable Playground page. The chain will appear in the dropdown.

### Programmatically
```python
from <package_name>.chains.my_chain import get_chain

chain = get_chain()
result = chain.invoke({"input": "test"})
print(result)
```

## Common LCEL Patterns

### Chain with Multiple Steps
```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Chain with Branching
```python
from langchain_core.runnables import RunnableBranch

chain = RunnableBranch(
    (lambda x: "math" in x["topic"], math_chain),
    (lambda x: "code" in x["topic"], code_chain),
    default_chain,
)
```

### Chain with Structured Output
```python
from pydantic import BaseModel

class Answer(BaseModel):
    reasoning: str
    answer: str
    confidence: float

chain = prompt | llm.with_structured_output(Answer)
```

### RAG Chain
```python
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.embeddings_factory import get_embeddings_store

store = get_embeddings_store()
retriever = store.as_retriever(search_kwargs={"k": 4})

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | def_prompt(
        system="Answer based on the context provided.",
        user="Context: {context}\n\nQuestion: {question}",
    )
    | get_llm()
    | StrOutputParser()
)
```

## Registration Options

```python
register_runnable(
    RunnableItem(
        tag="Category",          # Groups chains in the UI
        name="Display Name",     # Unique name for the chain
        runnable=get_chain,      # Factory function returning a Runnable
        examples=[               # Example inputs for the playground
            Example(query=["input 1"]),
            Example(query=["input 2"]),
        ],
    )
)
```
