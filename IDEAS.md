# Better Markdown Convertion

There are many solutions to convert documents  to markdown. We want to refactor our toolkit to make easier to integrate new one.   

Today we support Mistral OCR (genai_tk/workflow/loaders/mistral_ocr.py, genai_tk/workflow/markdownize/mistral.py), markitdown, edgeparse, and a custom spreadsheet parser (genai_tk/workflow/markdownize/converters.py). Markdown handling is spread over the code...  refactorinf is needed.

We want to support additional solutions : 
1/ LightOnOCR :  https://developers.lighton.ai/api-reference/parse/parse-a-document-to-markdown (we have the API key in the .env file)
2/ Anydoc : https://github.com/firecrawl/anydoc
3/ LLM  : use an LLM selected by a the langchain based LLM factory. Write the prompt and call the LLL async and in batch.  Expect that the given file format is supported by the LLM/provider - but have a clean message if error. 



Use our classical approach : an abstract class, instance per solution with contructor matching main solution parameters, and a YAML file to define common configuration, with a field pointing to the class. 

The converters should run async. If a batch mode is available (as with Mistral OCR), make it configurable.
Have an abstract method that return the list of supported file extensions for each solution. 


You could reuse and refactor genai_tk/default_config/markdownize.yaml, but be more flexible. I suggest selecting the converter by checking an ordered list of pathspecs.  Maybe a 'convertor selector' entry might be interesting ?  I let  you evaluate. 

Put core code in a new dir under genai_tk/extra.

Lazy import the necessary Python package. 
Update Prefect tasks and workflows. 
Update tests, docs, README and skills. 



Think first, ask question, suggest alternatives, etc. 





https://developers.lighton.ai/api-reference/parse/parse-a-document-to-markdown


We want to 


Status: 

~/prj/ekg-atos ->   cli kg create one_rainbow 
~/prj/rfq_pricing -> cli docgraph build $ONEDRIVE/prj/RFQ_pricing/RFQ_zipped/Alko.zip 


##  Anydoc
https://github.com/firecrawl/anydoc/blob/main/python/README.md 



# Simplify Integration
/home/tcl/prj/genai-tk/docs/design/deepagents-deerflow-langgraph-unification.md


```bash
# Markdownize a zip of raw RFQ documents directly — no separate unzip/office2pdf step
uv run cli workflow run markdownize --set sources=./RFQ.zip --set md_output_dir=./out/md

# genai-graph's doctree build always markdownizes first, then ingests into the tree DB
cli doctree build ./RFQ.zip --db ./data/kg/tree.db --profile fast

# Re-run just the Markdown conversion (and everything downstream of it)
cli doctree build ./RFQ.zip --db ./data/kg/tree.db --force md

# Re-parse the graph only, reusing the cached Markdown (no reconversion)
cli doctree build ./RFQ.zip --db ./data/kg/tree.db --force graph
```


# Pydantic
Replace @dataclass  by pydantic object
# TOC

We want to implement commands and Prefect tasks to create a table of content (TOC) from a Markdown document, and tools for agents
Inspiration is : 
- https://pageindex.ai/blog/pageindex-intro 
- https://github.com/VectifyAI/PageIndex/blob/main/pageindex/page_index_md.py
- https://github.com/VectifyAI/PageIndex/blob/main/examples/agentic_vectorless_rag_demo.py 


 Take inspiration of PageIndex parameters, but use our own convention to select the LLM, the class, etc. 

One difference with pageindex-intro  is that we want to create TOC from several Markdown files (typically in a  dir or a zip ), like in genai_tk/workflow/prefect/flows/merge_markdown_flow.py




First implement in genai-tk a simple workfow callable from 'cli workflow run' to create TOC from fiven markdown files. 
/home/tcl/prj/genai-tk/genai_tk/workflow

STATUS (2026-08-19): implemented in genai-graph instead of genai-tk — it leverages
the existing Document/MarkdownSection graph (hash-keyed, already has the heading
hierarchy) rather than a standalone JSON tree. See
`genai_graph/kg/document_graph/summarize.py`, `cli docgraph summarize`, and
`docs/document-graph.md` (genai-graph repo) for the implementation.

# LLM prompt caching (provider-side)

genai-tk has no provider-side *prompt* caching — `LlmCache` (`genai_tk/core/cache.py`)
is LangChain's exact-match response cache (SQLite/memory, keyed on the full prompt
string), which only helps identical re-runs. There is no support for a shared
*prefix* being cached across many different calls (e.g. summarizing N sections of
the same document, each call sharing the same long document context).

Providers that support this, and how:
- OpenAI: automatic prefix caching above 1024 tokens, no code changes needed, ~50%
  discount on the cached prefix.
- Anthropic: explicit `cache_control: {"type": "ephemeral"}` blocks in the message
  content, 5-minute TTL, ~25% write premium / ~90% read discount.
- Gemini/Mistral/EdenAI-proxy/local models: no equivalent today.

Would benefit any per-item-over-shared-context workload (document summarization,
batch classification/extraction over one big context, multi-turn agent scratchpads).
Needs a provider-agnostic API in `LlmFactory`/`get_llm()` that no-ops on providers
without support, rather than raising.

Raised while designing genai-graph's Document Graph summarization (`cli docgraph
summarize`): considered but not adopted a per-section LLM call design because of
this gap — see `genai_graph/kg/document_graph/summarize.py` docstring.


Then integrate it in ...


# More Harness
- Langchain coding harness + TUI
- Nvidia harness ? 
- Custom TUI made from the one in LC + Deerflow ?  

# Image in Markdown / Mistral
- include_image_base64=True 
- image_min_size = 40_000 
Saves extracted base64 encoded images into an _images subdirectory within the output folder for each PDF and updates markdown links to point to these local files.

- ?? add parameter confidence_scores_granularity = "page"

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


# Artifect
https://docs.prefect.io/v3/concepts/artifacts 

# SQL
Find/code a replacement for langchain_community.utilities.sql_database

nai_tk/core/cache.py


# tokenization 
use https://github.com/chonkie-inc/tokie 
(can remove tokenizers  - 3MB)



# Anonymimisation / LLM Routing demo
Create a Streamlit app that demonstrate features 
examples/notebooks/anonymize_rag_pipeline_demo.ipynb
examples/notebooks/middleware_anonymization_demo.ipynb

- The user select a short text among several prompt you have created, with different level of sensitivity
- it can either anonymize the prompt, or send it to a safe LLM, or both
- After submition, the possibly anomyzized text is displayed, and the destinated LLM, and some context informarion to explai  the choice
- The result returned with LLM is displayed
- the user can visualize the configurarauon and oyther information to understand how it works







# Around Agents

- Develop classical Deep Agents use case  , to run without too much change  (skill,  toools, MCP, ..) either in Deer-flow, Deeppagent-cli and our Langchain generic agent : research agent, coder agent, DB expert agent, etc...  
    - Test with several consiguration (sandbox, LLM, ...)
    - See https://github.com/langchain-ai/deepagents/tree/main/examples/  and Deer-flow 

- Implement Sub-agents  in our generic Langchain agent YAML config file

- Develop a better Anonimization Middleware, based on our Presidio extension and langchain build-in middleware

- Improve or replace our Rich based CLI by a Textual based one, inspired by deep-agent-cli 

- Implement an API for our agents, inspired by the one in Deep-Flow, so we could reuse its front-end to quick start a project

 - Integrate open-code in similar way than deep-flow, lanchain, deepagents  (using https://github.com/anomalyco/opencode-sdk-python)


## Other  

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


# CLI
cli workflow run baml_extract --set base_dir="$ONEDRIVE/prj/RFQ_pricing" --set output_dir="$ONEDRIVE/prj/RFQ_pricing/out"  --set function_name=ExtractRUFacts   --set pathspecs='["MERGED.md"]' --set llm=gpt5-mini@edenai --force

