# Routing Middleware
Update the routing middleware so it can use a decision model to acess whether a prompt contains sensitive data and cannot be anonimyzed. 
SensitivityScorer can become an ABC. 
Implement a DecisionModelSensitivityScorer.
Same with  a FilePathSensitivityScorer - use pathspec list instead of glob syntax
Make them chainable -> stop when high level of sensitificty is reached.


# Decision Models
We want to use easily decision models (also called system one models) : TypeSafe Jev, but also Cloudflare Chef, ..
We we restrict today to the one provided by OpenRouter : 
https://openrouter.ai/models?output_modalities=decisions

But that might change, and we could later implement TypeSafe System One API, or OpenRouter Decision API, or else.
We will start with Jev and Cloudflare Chef-flash.
You can have a look at : https://docs.langchain.com/oss/python/integrations/providers/typesafe#quickstart

(It might be a good idea to reuse their classes - you see)

Model.dev does no include these models, so we need a YAML file for their config (ex max tokens, capabilities, ...), like for embeddings and specials LLM.  And we need a factory to be independant of the model and the provider. 

Ensure methods to call these models use dict or (better) Pydantic models as input/output. I think all decision models will accept the same, but that need some verificatio?

A classical Langchain BaseChatModel can indeed be used as a Decision Model (just more expensive).  Provide a way to build a DecisionModel from a BaseChatModel.

Try to provide a  cli command "cli core classifier" to check it works for simple case. 

Here the OpenRouter API
import requests
import json

# The model answers narrow, typed questions about the state. Your code owns the workflow.
response = requests.post(
  url="https://openrouter.ai/api/alpha/decisions",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "Content-Type": "application/json",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-OpenRouter-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "cloudflare/clef-flash",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
      "is_urgent": {
        "type": "noul",
        "instructions": "Does this message convey urgency?",
        "criteria": {
          "true": "Explicitly time-sensitive",
          "false": "No urgency expressed"
        }
      },
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, refunds",
          "technical": "Bugs, outages, integrations",
          "sales": "Pricing, upgrades, new accounts"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  })
)

answers = response.json()["answers"]
# noul is a probability from 0 (no) to 1 (yes); choice and score carry the full distribution.
print(answers["is_urgent"]["noul"])
print(answers["department"]["choice"], answers["department"]["probabilities"])
print(answers["frustration"]["score"])

if answers["is_urgent"]["noul"] > 0.8 and answers["department"]["choice"] == "billing":
  pass  # escalate_to_billing(...)

but you can also have a look at :
https://docs.typesafe.ai/concepts/how-to-build-with-system-one


Search on internet on the best approach to abstracy call to these nex decision model, thing about a design (with maintenability in mind : that will likely evolve), and propose a plan.




Nemo: 
https://docs.nvidia.com/nemo/relay/integrate-into-frameworks/adding-scopes




# OPA Security ? 

https://github.com/open-policy-agent/opa

Don't confuse with: 
https://github.com/proishan11/open-agent-policy/blob/main/examples/minimal-agent/langchain_agent.py



# LLM prompt caching (provider-side)

https://openrouter.ai/docs/guides/best-practices/prompt-caching

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

# Refactor Retriever 
We want to completly refactor the RAG processing part of the toolkit, to ba able to deal with more complex use cases, backends and configuration. We want notably able to levearge the capabilities of hybrid rag of the zvec lib (genai_tk/core/vector_backends/zvec.py ), in addition of current use cases with PostgreSQL, ZeroEntropy, and vector store + bm25 +  reranker. 

Our idea is this one : 
- ManagedRetriever should become an abstract class , with core abstract methods such as aquery, aadd_documents, adelete_colection, ...It could inherit langchain Retriever base class, or have a get_retriever method that returns one. 
- We could keep the concept of RAGToolFactory - to get a tools usable from an agent
- Remove SQLRecordManager and replace caching with a configurable mechanisme : either we can query the vector-store to check that a hash of the chunk + medatata + embeddings model has been inserted, or we put that information in a KV store built with py-key-value (already used in the project). 
- Each concrete ManagedRetriever (with pgvecor, zvec, vertor-store+bm25s, ...) should at least be able to do hybrid search (vector + full text search) with reranking (either RRF or given reranker model). Adapt configuration and possible extra feature to the actuel implementation (read the doc ! )

Adapt the Prefect workflows and examples accordingly.honkie


 
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

