"""Simple joke chain — demonstrates LCEL (LangChain Expression Language) composition."""

from genai_tk.core.chain_registry import Example, RunnableItem, register_runnable
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.prompts import def_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable


def get_chain(config: dict | None = None) -> Runnable:
    """Create a joke chain: prompt | llm | parser."""
    llm = get_llm()
    prompt = def_prompt(
        system="You are a witty comedian. Keep jokes short and clever.",
        user="Tell me a joke about {topic}.",
    )
    return prompt | llm | StrOutputParser()


# Register so the chain appears in the webapp Runnable Playground
register_runnable(
    RunnableItem(
        tag="Example",
        name="Joke",
        runnable=get_chain,
        examples=[Example(query=["programming"])],
    )
)
