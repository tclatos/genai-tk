# Genai-Tk Evolution Ideas - Roadmap candidates - 


- Merge  genai_tk/extra/retrievers into  genai_tk/core/retrievers.  Simplify file structure. 
- move extra/rag to core/rag
- remove /home/tcl/prj/genai-tk/genai_tk/extra/loaders/markdown_loader.py
Update documentation and tests




# Splitter 

We want to improve the text chunking in RAG.  Today when  calling 'cli rag add-files', it calls _chunk_file_content that select among 2 Chonkie provided "BaseChunker". 
A more flexible approach using Factories defined in YAML would be better : define in a YAML file a list of chunkers, with for each a class (qualified name, descendant of Langchain TextSplitter) with its arguments (size, overlapp, ...),as we do in other cases.
Start with the classic Langchain RecursiveCharacterTextSplitter and a Chonkie Markdown one based on markdown_chunking.py, and a Chonkie recursive, with common parameters. 

Update genai_tk/extra/rag/markdown_chunking.py so it inherit TextSplitter and returns Langchain Documents objects and (with "start_index" metadata set , and number of token, even it's not a standard information for Langchain ) . Have something generic so other Chonkie chunkers can be used in LangChain ecosystem/


Update cli rag add-files to select the chunker. Have an "auto" choice so the chunker is selected according to the extension of the file (md => Markdown, ...) 

Have a rag.yaml for these definition. Move the embeddings_store and retrievers definition there (from baseline.yaml). Update app_conf.yaml accordingly.

Update /home/tcl/prj/genai-graph/genai_graph/webapp/pages/demos/kg_lineage.py so its take a chunker definition.


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



