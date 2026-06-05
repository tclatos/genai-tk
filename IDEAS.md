# Goal
I want to improve skills and scafolding. A refactoring is likely required.
Today : 
1/  when running command  'cli skills list', many skills appears, but they are in fact in 2 categories : the skills usefull to build the app or the agent from genai-tk or a  scafolded project (like genai-graph and the recent rfq_pricing) (typically  genai-tk-*), and skills used by created agents - that are mainly examples.   We need to have more clear for the user, and also more optimized when loading skills from a coding agent or a created agent. 
2/ In scafolding template there are directories for Cursor and Windsurf. Try to avoid, and prefer a mechanism that 
allo installing in any IDE . It'n necessealy done by us : we could just tell the user what command to enter.
3/ We could also help the user to install development skills for the Python package use use : Langchain, BAML, Prefect, ... (so he don't have to search)
4/ Instead of selecting templates (rag, agent, ...), the user might selection optional components.  For today, make just 1/ Deer-flow 2/ aio-sandox  optionals because they are heavy 
5/  There's an error with current scafolding.  TRace and analysis at the end of this message.
6/ Deer-flow integration is broken
7/ Scafolding Template are broken

# Ideas to investigate
Think about a refactoring.  Here ideas I have : 
- I like the cli skill idea command.  We could get inspiration for better design with Strealit analysys https://github.com/streamlit/streamlit/blob/develop/specs/2026-05-11-streamlit-skills-cli/product-spec.md , and https://github.com/tiangolo/library-skills ;  - Maybe leveraging npx skills might also be usefull .
- The 'cli init' idea could also be more complete 

Take into account the deer-flow refactoring analysis provided later. 

Think, ask questions, suggest improvements. Propose a plan. Compatibility  is not required : anything can be changed. Maintenability is key.

------
# Analysis for Deer Flow Improvement

## DeerFlow Integration: Current State & Better Approach

### Why it's currently broken

The integration works by cloning the repo and adding `DEER_FLOW_PATH/backend/packages/harness` to `sys.path`, then importing `deerflow.client.DeerFlowClient`. Several things break this:

**1. Installation approach is fragile (`_install_deer_flow_backend` in commands_init.py)**

The function reads deps from `backend/pyproject.toml` and runs:
```bash
uv pip install -e backend/packages/harness <all-deps>
```
But `backend/pyproject.toml` itself lists `deerflow-harness` as a **uv workspace dependency** (`deerflow-harness = { workspace = true }`). `uv pip install` doesn't understand workspace sources — it looks for `deerflow-harness` on PyPI (it doesn't exist there), so this may fail depending on resolution order.

**2. Upstream architecture changed significantly**
- Commit `#1131` split the backend into `harness` (`deerflow.*` modules) and `app` (`app.*` modules) ~3 months ago — the embedded_client.py already supports both layouts, but version drift causes API mismatches
- `langgraph.checkpoint.sqlite` import (line ~430) was moved to a separate `langgraph-checkpoint-sqlite` package in recent LangGraph 1.x
- `DeerFlowClient.__init__` signature continues to evolve; the `inspect`-based kwarg detection helps but can't catch renamed params

**3. Version conflicts with genai-tk**
DeerFlow's `deerflow-harness` pins `langchain>=1.2.15`, `langgraph>=0.4.8` etc., which can conflict with genai-tk's own LangChain/LangGraph pins, especially since both are installed into the same venv via `uv pip install`.

---

### Recommended: install `deerflow-harness` as a proper uv Git dependency

The `deerflow-harness` package at `backend/packages/harness/` is a standalone, properly structured Python package (`name = "deerflow-harness"`). uv supports installing from a Git subdirectory:

```bash
uv add "deerflow-harness @ git+https://github.com/bytedance/deer-flow@main#subdirectory=backend/packages/harness"
```

**Advantages:**
- `DEER_FLOW_PATH` env var no longer needed — code is installed normally
- `uv lock --upgrade-package deerflow-harness` handles updates cleanly
- No `sys.path` manipulation needed (import `deerflow.client` directly)
- Works from any IDE or environment without extra setup steps
- Pin to a commit hash for stability: `@abc1234#subdirectory=...`

**In pyproject.toml**, the `deer-flow` optional group would become:
```toml
[project.optional-dependencies]
deer-flow = [
    "deerflow-harness @ git+https://github.com/bytedance/deer-flow@main#subdirectory=backend/packages/harness",
]
```

**`cli init --deer-flow` would simplify to:**
```bash
uv add "deerflow-harness @ git+https://github.com/bytedance/deer-flow@<PINNED_COMMIT>#subdirectory=backend/packages/harness"
```
And updating becomes:
```bash
uv add "deerflow-harness @ git+https://github.com/bytedance/deer-flow@main#subdirectory=backend/packages/harness"
```

The `_ensure_deer_flow_on_path()` and all `sys.path` manipulation in embedded_client.py become unnecessary — just `from deerflow.client import DeerFlowClient` directly.

---

### What to do with `cli init --deer-flow`

Keep `--deer-flow` as an optional flag (aligned with IDEAS.md point 4), but change its implementation:

1. **Install**: runs `uv add deerflow-harness @ git+...@main#subdirectory=backend/packages/harness`
2. **Update**: same command — `uv add` re-resolves to latest
3. **Pin**: optionally pin to a known-good commit (add a `--deer-flow-pin <SHA>` option)
4. **No clone needed**: no local git clone, no `DEER_FLOW_PATH` — the package is in the venv like any other dependency
5. **Skills**: DeerFlow skills are inside the harness package (at `deerflow/skills/` or discoverable), so no need to mount from a separate clone path

The skills directory (skills referenced in deerflow.yaml) would need to either be bundled in the harness or fetched separately — worth checking if `deerflow-harness` includes them.

---

### Quick summary

| Current | Better |
|---|---|
| `DEER_FLOW_PATH` env var → git clone → `sys.path` hack | `uv add deerflow-harness @ git+...#subdirectory=backend/packages/harness` |
| `uv pip install` with workspace dep conflict | Standard `uv` dependency management |
| Manual `cli init --deer-flow --force` for updates | `uv add ... @main` or `uv lock --upgrade-package` |
| Legacy/modern layout detection code | Removed entirely (import directly) |
| Skills from clone path | From installed package |


# Current issue  with scaffolding
fr23439@DESKTOP-8PNA2NI:~/my-genai-project$ uv add git+https://github.com/tclatos/genai-tk@main
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtual environment at: .venv
Resolved 332 packages in 541ms
  × Failed to build `genai-tk @ git+https://github.com/tclatos/genai-tk@7a84e27398ab78cbd19c664f6f8c9c0236844f52`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `hatchling.build.build_wheel` failed (exit status: 1)

      [stderr]
      Traceback (most recent call last):
        File "<string>", line 11, in <module>
        File "/home/fr23439/.cache/uv/builds-v0/.tmpVkkQex/lib/python3.12/site-packages/hatchling/build.py", line 58,
      in build_wheel
          return os.path.basename(next(builder.build(directory=wheel_directory, versions=["standard"])))
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File
      "/home/fr23439/.cache/uv/builds-v0/.tmpVkkQex/lib/python3.12/site-packages/hatchling/builders/plugin/interface.py",
      line 157, in build
          artifact = version_api[version](directory, **build_data)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/fr23439/.cache/uv/builds-v0/.tmpVkkQex/lib/python3.12/site-packages/hatchling/builders/wheel.py",
      line 521, in build_standard
          record = archive.add_file(included_file)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/fr23439/.cache/uv/builds-v0/.tmpVkkQex/lib/python3.12/site-packages/hatchling/builders/wheel.py",
      line 131, in add_file
          with open(included_file.path, "rb") as in_file, self.zf.open(zip_info, "w") as out_file:
                                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/fr23439/.cache/uv/builds-v0/.tmpVkkQex/lib/python3.12/site-packages/hatchling/builders/wheel.py",
      line 78, in open
          raise ValueError(message)
      ValueError: A second file is being added to the wheel archive at the same path:
      `genai_tk/default_config/Makefile.template`.

      The most likely cause of this is an entry in the `tool.hatch.build.targets.wheel.force-include` table. See:
      https://hatch.pypa.io/1.8/config/build/#forced-inclusion



      hint: This usually indicates a problem with the package or the build environment.
  help: If you want to add the package regardless of the failed resolution, provide the `--frozen` flag to skip
        locking and syncing.

Voici l'analyse de GPT :
Je pars de l’erreur de build elle-même: genai_tk/default_config/Makefile.template est inclus deux fois dans la wheel. Je vais confirmer, à partir du dépôt, quelles entrées de packaging provoquent ce doublon et en déduire la cause précise plutôt qu’une hypothèse générale.
L’erreur n’est pas côté uv ni côté votre projet. C’est un bug de packaging dans genai-tk.
Le point clé est ici :
ValueError: A second file is being added to the wheel archive at the same path:
genai_tk/default_config/Makefile.template
Ça signifie que, pendant la construction de la wheel, Hatchling essaie d’ajouter deux fois le même chemin interne. La cause la plus probable est la combinaison suivante dans le dépôt :
le package genai_tk contient un lien symbolique default_config qui pointe vers config
le build force aussi l’inclusion de config/ vers genai_tk/default_config/
Résultat : Hatchling voit une première fois Makefile.template via le package lui-même, puis une seconde fois via le force-include, et il bloque.




# use key-value store factory 
use py-key-value-aio
https://strawgate.com/py-key-value/adapters/#basemodeladapter
https://strawgate.com/py-key-value/adapters/#pydanticadapter



# markdown
all option to generate HMTMl with  Edgeparse
 
Implement a "jsonify" version with liteparse v2

# Artifect
https://docs.prefect.io/v3/concepts/artifacts 

# Skill
inspired by https://github.com/streamlit/streamlit/pull/15116 ? 

# scafolding
- suggest after init to install BAML : uv run baml-cli init --dest <PATH>   

# tokenization
use https://github.com/chonkie-inc/tokie 


Create a command calling a worflow that 
- cli workflow run baml_extract --set base_dir="$ONEDRIVE/prj/RFQ_pricing" --set output_dir="$ONEDRIVE/prj/RFQ_pricing/out"  --set function_name=ExtractRUFacts   --set pathspecs='["MERGED.md"]' --set llm=gpt-oss-120@openrouter --force

# BAML
Develop Workflows to generate structured output with BAML, usable through the 'cli workflow run" command, and composable . Update  commands 'cli  baml extract' and 'cli  baml run' so they call  'cli workfow'  (they will likely be depreciated later)

# Model to Table CSV
Create a Workflown that takes a JSON file, a Pydantic model and a list of keys, and generate a table in CSV or Excel format. 
Then create a workflox that  combine it with the BAML workflow previously created.

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

