# NLP
We have Spacy and nlp code used in different part of the toolkit. It's time to refactor.
Move genai_tk.workflow.anonymization.presidio_detector (and possiblye other part of the module)  to genai_tk.extra.nlp  and refactor to have better control of Spacy for other use cases other than PII (ex: Hybrid search, classifier), or to select language, models, or else ...

Move also and refactor /home/tcl/prj/genai-tk/genai_tk/utils/spacy_model_mngr.py 


Ensure strong typing, and correct error handling  if nlp feature is not enabled

Check that PPI works for French - or raise a clear message is the model is not loaded


Refactor genai_tk/agents/langchain/middleware/presidio_detector.py and related files.
Consider refactoring genai_tk/agents/langchain/middleware/sensitivity_scorer.py : the sensitivity scorer could be a quite reusable classifier (that we would update). Think about that (and having some form of classifier factory). But its a secondary objective.

Refactor genai_tk/core/retrievers/bm25.py and/or genai_tk/core/factories/retriever_factory.py  to have a better Spacy support (more configurable, single place, ...).

Refactor tests. 
We should have a better design after these changes !  

Think, ask questions, suggest improvements, ..



# Refactor Retriever 
We want to completly refactor the RAG processing part of the toolkit, to ba able to deal with more complex use cases, backends and configuration. We want notably able to levearge the capabilities of hybrid rag of the zvec lib (genai_tk/core/vector_backends/zvec.py ), in addition of current use cases with PostgreSQL, ZeroEntropy, and vector store + bm25 +  reranker. 

Our idea is this one : 
- ManagedRetriever should become an abstract class , with core abstract methods such as aquery, aadd_documents, adelete_colection, ...It could inherit langchain Retriever base class, or have a get_retriever method that returns one. 
- We could keep the concept of RAGToolFactory - to get a tools usable from an agent
- Remove SQLRecordManager and replace caching with a configurable mechanisme : either we can query the vector-store to check that a hash of the chunk + medatata + embeddings model has been inserted, or we put that information in a KV store built with py-key-value (already used in the project). 
- Each concrete ManagedRetriever (with pgvecor, zvec, vertor-store+bm25s, ...) should at least be able to do hybrid search (vector + full text search) with reranking (either RRF or given reranker model). Adapt configuration and possible extra feature to the actuel implementation (read the doc ! )

Adapt the Prefect workflows and examples accordingly.honkie

## Ladybug embeddings

We want to store in Ladybug the embeddings of some kinf of documents.  



 
# huggingface
 Check it accetp streaming, .... 
https://docs.langchain.com/oss/python/integrations/llms/huggingface_endpoint
Voir StreamingStdOutCallbackHandler

Factory de provider ?

# markdown
all option to generate HMTMl with  Edgeparse
 
Implement a "jsonify" version with liteparse v2

# Artifect
https://docs.prefect.io/v3/concepts/artifacts 

# SQL
Find/code a replacement for langchain_community.utilities.sql_database

nai_tk/core/cache.py


# tokenization 
use https://github.com/chonkie-inc/tokie 
(can remove tokenizers  - 3MB)


# TOC

We want to implement commands and Prefect tasks to create a table of content (TOC) from a Markdown document. 
Inspiration is : 
- https://pageindex.ai/blog/pageindex-intro 
- https://github.com/VectifyAI/PageIndex/blob/main/pageindex/page_index_md.py
- https://github.com/VectifyAI/PageIndex/blob/main/examples/agentic_vectorless_rag_demo.py 


 Take inspiration of PageIndex parameters, use our own convention to select the LLM, the class, etc. 

First implement in genai-tk a simple workfow callable from 'cli workflow run' to create TOC from fiven markdown files. 
/home/tcl/prj/genai-tk/genai_tk/workflow

Then integrate it in ...

# Anonymimisation / LLM Routing demo
Create a Streamlit app that demonstrate features 
examples/notebooks/anonymize_rag_pipeline_demo.ipynb
examples/notebooks/middleware_anonymization_demo.ipynb

- The user select a short text among several prompt you have created, with different level of sensitivity
- it can either anonymize the prompt, or send it to a safe LLM, or both
- After submition, the possibly anomyzized text is displayed, and the destinated LLM, and some context informarion to explai  the choice
- The result returned with LLM is displayed
- the user can visualize the configurarauon and oyther information to understand how it works









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
Refactor /workflow/loaders/markdown_loader.py with improvement from /workflow/rag/markdown_chunking.py.
Keep a LangChain interface (ie Document + metadata instead of ChunkInfo - as TypedDict if possible - and inherit BaseLoader ).
Replace code in genai-graph that uses markdown_chunking with the LangChain compatible loader/splitter. 
Add test cases.
Consider using PageIndex (of be inspired by) to have a TOC and a better structure (https://github.com/VectifyAI/PageIndex/blob/main/pageindex/page_index_md.py)


###  RAG
Refactor totaly  /home/tcl/prj/genai-tk/genai_tk/tools/langchain/rag_tool_factory.py .  
The created LangChain tool should behave like the 'query' command in /home/tcl/prj/genai-tk/genai_tk/workflow/rag/commands_rag.py, ie accept a query string and an optional metadata filter in JSON. 
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



# CLI
cli workflow run baml_extract --set base_dir="$ONEDRIVE/prj/RFQ_pricing" --set output_dir="$ONEDRIVE/prj/RFQ_pricing/out"  --set function_name=ExtractRUFacts   --set pathspecs='["MERGED.md"]' --set llm=gpt5-mini@edenai --force

