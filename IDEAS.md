#  just commands 
Create a just command that takes a path to a directory name as argument and create a new scaffolded project 
(create the dir if not exists, uv init, uv add genai_tk, cli init, ... )

# Hakathon

ok. The goal was to prepare an hackthon for pen-testers leaning agenic AI. 

1/  rename  prjtest  as pentest-demo 
2/ Write a document (for the teacher) to guide the students. Start with the steps descrived in /home/tcl/prj/genai-tk/IDEAS.md, and add additonal strps to make the created agents run with skills in a sandbox
2/ push the projet  in a new git hub repo (you can use gh). I will make it private later (so the student won't  cheat....)
3/ 
## First step

Create an agent 'pentester'. Use the DeepAgent harness.
- Write a tool factory** exposing nmap/nikto/whatweb/gobuster/ sqlmap as LangChain tools that run as host subprocesses.
- Add these tools to the agent.
- Add a CLI command group** (cli pentest ...) to manage the agent.
- test the agent with a simple prompt asking for a nmap localhost.
- Use skills.

## 2 
1/ use bkimminich/juice-shop as target for the pentests. Provide CLI commmand (in pentest group) to start/stop the docker image
2/  Write  a skill to perform  a simple penetration test on juice-shop using tools we have
3/ test it 

## 3
We want to use a docker sandbox for the tests. We will use the AIO sandbox called from DeepAgents. 
This combination has not been fully tested. You might need to update ganai-tk.
1/ install nmap, nikto, whatweb, gobuster and sqlmap 



export PRJ=prjtest; rm -rf $PRJ; mkdir $PRJ; cd $PRJ ; uv init; uv add git+https://github.com/tclatos/genai-tk@main; cli init --extra harnessing


sélection concrète, crédible et directement exploitable de :

🧱 MCP servers (ou assimilables)
🔌 Skills / tools agents

…organisés par scénario pentest réel (ça parlera immédiatement à ton audience).

🔴 1. Offensive / Pentest automation
🧠 MCP serveur “full-stack pentest”
🟢 HexStrike AI (MCP)

MCP server orchestrant 150+ tools de cybersécurité
multi-agents (reco, CVE, exploit, reporting)
cible :

bug bounty
CTF
vuln discovery



👉 très puissant pour hackathon (effet “wow”)
 [github.com]

🔌 Skills à intégrer

Nmap skill

scan réseau / ports / services


Metasploit skill

exploitation automatique


SQLmap skill

injection SQL


Gobuster / Amass

enumeration / attack surface



👉 ces outils sont standards du pentest (scan, exploitation, enum) [esecurityplanet.com], [cybersecur...tynews.com]

💡 Cas hackathon

“Build an autonomous pentest agent”


agent lance reconnaissance
choisit exploit
génère rapport


🔵 2. Web security / AppSec
🧱 MCP / gateway possible

MCP wrapper autour d’un scanner web (ZAP / Burp API)
ou simple serveur exposant APIs

🔌 Skills

OWASP ZAP skill

proxy + scan web (XSS, injection…) [esecurityplanet.com]


Nikto

vuln serveur web


w3af

audit + exploitation web




💡 Cas hackathon

“Agent web hacker”


prend URL cible
lance scan
exploite vuln simple
propose remediation


🟣 3. Network / infra analysis
🔌 Skills

Wireshark / tcpdump

analyse trafic réseau [esecurityplanet.com]


Nmap NSE scripts

détection vuln automatée [guptadeepak.com]




🧱 MCP idea

MCP server exposant :

capture logs réseau
alerting
historique




💡 Cas hackathon

“AI SOC analyst”


détecte anomalies dans logs réseau
propose attaque probable


🟠 4. Reverse engineering / malware
🔌 Skills

Radare / IDA wrapper

analyse binaire [cybersecur...tynews.com]


Apktool

reverse app Android [cybersecur...tynews.com]




💡 Cas hackathon

“Malware triage agent”


analyse fichier suspect
identifie comportements
produit rapport


🟡 5. Identity / AD attack
🔌 Skills

BloodHound (API wrapper)

graph AD attack paths [guptadeepak.com]


hashcat / John the Ripper

password cracking [esecurityplanet.com]




💡 Cas hackathon

“AD attack planner agent”


analyse AD dump
propose chaine d’attaque


⚫ 6. AI security (très intéressant pour ton positionnement)
🧱 MCP / Skills spécifiques AI
🟢 SkillSpector (NVIDIA)

scanner de skills agents
détecte :

prompt injection
malware
exfiltration
 [github.com]




🔐 Autres skills utiles

prompt injection tester
secrets scanner
dependency scanner

👉 car:

~26% des skills contiennent des vulnérabilités [explainx.ai]


💡 Cas hackathon

“Secure the agent ecosystem”


scanner des skills
détecter attaques
bloquer install


🧱 7. MCP “infrastructure” (très pédagogique)
Idées de serveurs MCP simples à builder
🟢 MCP Filesystem

accès fichiers
exfiltration possible

🟢 MCP GitHub

accès code + secrets

🟢 MCP Logs / SIEM

accès logs sécurité

📌 Important pour le hackathon :
👉 Un MCP donne accès à :

fichiers
APIs
cloud
👉 donc devient point unique de compromission [sentinelone.com]


🎯 Mapping rapide (ce que tu peux proposer aux équipes)













































CatégorieMCP / SkillPentest autoHexStrike MCPScan réseauNmapExploitMetasploitWebZAP / NiktoLogsWiresharkReverseRadare / ApktoolPasswordHashcatADBloodHoundAI securitySkillSpector

🔥 Conseils pratiques pour ton hackathon
✅ Donne :

1 agent de base
3–4 skills pré-packagés
1 MCP vulnérable

✅ Laisse :

ajouter leurs propres tools
créer nouveaux skills
modifier MCP server


🧠 Insight clé (important pour ton discours)
👉 Le point fondamental à faire passer aux pentesters :

“Les agents ne sont pas le sujet.
Les agents = un orchestrateur d’outils… et donc une surface d’attaque énorme.”

Car :

MCP connecte à tous les systèmes critiques [sentinelone.com]
les tools sont exécutés automatiquement par le LLM [cheatsheet....owasp.org]





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







# Around Agents
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

