# Genai-Tk Evolution Ideas - Roadmap candidates - 

# Test Notebooks
- extend cli test with : https://medium.com/codetodeploy/automating-jupyter-notebook-testing-a-liteweight-approach-f723273eeacf 
- have YAML config for tests
# LLM Router

Create  anonymizer and router Middlewares for LangChain agents, based on Presidio.   For the anonymization, get inspiration from /home/tcl/prj/genai-tk/genai_tk/extra/custom_presidio_anonymizer.py (quite old code - can be improved)   and build-in PII Midleware (https://github.com/langchain-ai/langchain/blob/87ba30f09773b8e9ec549841c57906f343b35ed8/libs/langchain_v1/langchain/agents/middleware/pii.py) (https://reference.langchain.com/python/langchain/agents/middleware/pii/PIIMiddleware.md)
For the routing, have a look at : /home/tcl/ext_prj/LLM_router/llm_router/middleware 

The idea is to have 2 Middleware taking as argument A Pydantic object to define the analyser work. 
For the anonymization it could includes field from the legacy YAML conf (config/demos/presidio_anonymization.yaml)
For the router, it could include 
    - the qualified path to a function (or a Class - you decide) that math match done in /home/tcl/ext_prj/LLM_router/llm_router/middleware/confidentiality.py , but with Presidio and/or spaCy.  It returns whether the message is sensitive and need to be send to a 'safe' llm, or not  - Provide one by default
    - A pydainc model to confure such function - Put there the equivalent of hard coded dict  in confidentiality.py.  Hard code it however for no - we will see later to have it it YAML.
    - an id or tag for the save llm
    - a list of file path pattern (glob style) 

This file paths are used to dedermine if the context extracted from a retriever (for RAG) need systematocally be sent to a safe LLM, whatever its content. To do that, my idea (to be checked) is a middleware wrapper around tools that checks is the tool return a list of Langchain Document object in the 'artifact' field of the ToolMessage returned after tool call, if if so collect in a set the 'source' key value in the metadata.    






# Better file spec
use  https://github.com/cpburnz/python-pathspec 



## Improve Embedders
-  hane new FastEmbedEmbeddings provider: https://reference.langchain.com/python/langchain-community/embeddings/fastembed/FastEmbedEmbeddings  ; https://qdrant.github.io/fastembed/examples/Supported_Models 
- add category (sparce, reranking, image, ..)

## Around Agents
- Test the AioSandboxBackend (taken from DeerFlow, and made compatible with Langchain protocol) to work with Deep agents and Deepagent-cli. 

- Develop classical Deep Agents use case  , to run without too much change  (skill,  toools, MCP, ..) either in Deer-flow, Deeppagent-cli and our Langchain generic agent : research agent, coder agent, DB expert agent, etc...  
    - Test with several consiguration (sandbox, LLM, ...)
    - See https://github.com/langchain-ai/deepagents/tree/main/examples/  and Deer-flow 

- Implement Sub-agents  in our generic Langchain agent YAML config file

- Develop a better Anonimization Middleware, based on our Presidio extension and langchain build-in middleware

- Improve or replace our Rich based CLI by a Textual based one, inspired by deep-agent-cli 

- Implement an API for our agents, inspired by the one in Deep-Flow, so we could reuse its front-end to quick start a project

 - Integrate open-code in similar way than deep-flow, lanchain, deepagents  (using https://github.com/anomalyco/opencode-sdk-python)


## Other  

###  Markdown loader
Refactor /extra/loaders/markdown_loader.py with improvement from /extra/rag/markdown_chunking.py.
Keep a LangChain interface (ie Document + metadata instead of ChunkInfo - as TypedDict if possible - and inherit BaseLoader ).
Replace code in genai-graph that uses markdown_chunking with the LangChain compatible loader/splitter. 
Add test cases.
Consider using PageIndex (of be inspired by) to have a TOC and a better structure (https://github.com/VectifyAI/PageIndex/blob/main/pageindex/page_index_md.py)


###  RAG
Refactor totaly  /home/tcl/prj/genai-tk/genai_tk/tools/langchain/rag_tool_factory.py .  
The created LangChain tool should behave like the 'query' command in /home/tcl/prj/genai-tk/genai_tk/extra/rag/commands_rag.py, ie accept a query string and an optional metadata filter in JSON. 
In the factory, we pass the name of the embedding store (to be used by EmbeddingsStore.create_from_config...) , 
tool name, tool descripton and default metadata filter  (to be merge with the one given when calling the tool).
look at /home/tcl/prj/genai-tk/genai_tk/tools/langchain/sql_tool_factory.py, that works.


 ## better LiteLLM support

- Refactor  get_litellm_model_name  so it works with our new LlmFactory and with more providers.
- Allow LiteLLM naming in complement to our own ( ex: uv run cli core llm -i 'tell me a jole' -m openrouter/google/openai/gpt-4.1-mini  )


## Hybrid search extension to genai_tk/core/embeddings_store.py
- use BM25S + Spacy (but configurable)


# Misc
Use https://github.com/GrahamDumpleton/wrapt for @once



