# Genai-Tk Evolution Ideas - Roadmap candidates - 

# Rag Retriever 

We want to improve RAG in our framework,  notably to facilitate Hybrid Search, that today can be done either with a vector store + bm-25 like + Ensemble retriever, or with PostgreSQL with hybrid search.
The genai_tk/core/embeddings_store.py is not perfect, has it does not nicely fit with an hybrid search definition. 

The idea I have is to use the factory pattern, used several time in the toolkit.  We could have factories that return a Langcchain retriever - that could be a simple vector store, or the curent "PostgreSQL with hybrid search", or an Ensemble retriever (vector store + bm25s), or new one (not necesseraly linked to a vector store).   Such factory should be recursive (ex: Esnsemble retriever factory) . They can be  configured in YAML. 

Such retrovers will usualy be accessed by a Langchain tool. There's already a genai_tk/tools/langchain/rag_tool_factory.py -  that can likely be improved. 


However, we need to take into accout that, before being retrieved, the data need to be stored.  Do the factories return object should also have the ability to store Documents with their metadata (like Langchain Vector Store). In case of ensemble retriever, the data (more precisely the index) should be sored in bith vector store and BM25 like. 
We want also to avoid recalculation - typically using Langchain record_manager feature (as today).


We want also to facilitate tests and deployement with Postgres, using  pgembed (https://github.com/Ladybug-Memory/pgembed). So have a nice way to select our way to access Postgres.  Be aware that some tools need async version, other can't (so there is a hack somewhere)

Analyse the code, including the cli rag commands, and propose a plan to refactor, simplify and easier to maintain and update. You can change everything : don't care about legacy code.  Plan to update tests and cod, and to create one or two notebooks to illustre how things work.

Ask questions, suggest improvement or alternative approach.










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



