
# Tracing

Improve monitoring and tracing support in the Tk.  Today, only Langsmith is supported (impliclt - through Langchain).
We want the uset to be able to select its tracing infra, with at least 2 framework possible: 
- Langsmith
- LangFuse
and possibly with support of 
- OpenTelemetry (but might be done with langfuse ? )
- Most other telemetries platforms  (have provision for it)

Also, we would like that all (or most) tools we use able to send traces in Langchain or Langfuse: 
- Lanchain 
- Deer flow
- SmollAgents
- BAML
- LiteLLM 

Last but not least, we would like our own basic logging, in local JSONL files, with date, session, prompts (wrapped), number of tokens, cost.. (whenever possible)  . It should run in parallel of the more sophisticated one. 

AFAIK, 
  - Langchain supports Langsmith nativelly, LangFuse (https://docs.langchain.com/oss/python/integrations/providers/langfuse), OpenTelemetry (https://docs.langchain.com/langsmith/trace-with-opentelemetry, https://www.langchain.com/blog/end-to-end-opentelemetry-langsmith)

  - Deer flow supports LangSmith  and LangFuse (https://github.com/bytedance/deer-flow/blob/d133b1119a955564c80b83b8bbe25aef351629f8/backend/README.md?plain=1#L330 ) 

  - BAML seems a more comlicatec case . There are https://docs.boundaryml.com/guide/baml-advanced/collector-track-tokens that can be used for our basic local logging, and be extandes later to LanGuse, ..

  - SmolaAgantes support OpenTelemetry (https://huggingface.co/docs/smolagents/tutorials/inspect_runs). LangFuse can be used, too (https://langfuse.com/integrations/frameworks/smolagents)

- LiteLLM supports LangFuse (https://langfuse.com/integrations/frameworks/litellm-sdk ) , LangSmith (https://docs.litellm.ai/docs/observability/langsmith_integration), OPenTelemety (https://docs.litellm.ai/docs/observability/opentelemetry_v2), ...

Follow the usual pattern
 - YAML file to define monotorigns configurations 
 - Selection of a given config through a variable 
 - cli commands for common operations : start the server (in case of Langfuse local one in docker), open UI, ...   (use  langfuse Python API), print nicely last events in local file, ...  





 
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


## Ladybug embeddings

We want to store in Ladybug the embeddings of some kinf of documents.  







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

