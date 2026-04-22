"""Hello Agent — a simple Streamlit page demonstrating a ReAct agent with chat UI.

This page shows how to build an interactive agent interface using genai-tk components.
Customize or copy this file as a starting point for your own agent pages.
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from streamlit import session_state as sss

from genai_tk.core.llm_factory import get_llm
from genai_tk.webapp.ui_components.llm_selector import render_llm_selector

st.set_page_config(page_title="Hello Agent", page_icon="🤖", layout="wide")
st.title("🤖 Hello Agent")
st.caption("A simple ReAct agent demo — edit this page to build your own!")

# ── Sidebar: LLM selection ──────────────────────────────────────────────
with st.sidebar:
    llm_id = render_llm_selector(key="hello_agent_llm")


# ── Define a sample tool ────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use Python syntax, e.g. '2 + 2' or '3 ** 4'."""
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ── Session state ───────────────────────────────────────────────────────
if "hello_messages" not in sss:
    sss.hello_messages = [AIMessage(content="Hello! I can answer questions and do math. Try me!")]

# ── Display chat history ────────────────────────────────────────────────
for msg in sss.hello_messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── Chat input ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything..."):
    sss.hello_messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = get_llm(llm_id or "default")
            agent = create_react_agent(llm, tools=[calculator])
            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
            )
            answer = result["messages"][-1].content
            st.markdown(answer)

    sss.hello_messages.append(AIMessage(content=answer))
