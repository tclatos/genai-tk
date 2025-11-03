"""Web search tools for AI applications.

This module provides tools for performing web searches, with support for multiple
search providers including Tavily and DuckDuckGo. The tools are designed to be
used in AI workflows where external information retrieval is needed.

Key Features:
- Automatic provider selection based on API key availability
- Configurable result limits
- Unified interface for different search backends
"""

import os

from langchain_core.tools import tool

# def internet_search(
#     query: str,
#     max_results: int = 5,
#     topic: Literal["general", "news", "finance"] = "general",
#     include_raw_content: bool = False,
# ):
#     """Run a web search"""

#     from tavily import TavilyClient

#     tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
#     return tavily_client.search(
#         query,
#         max_results=max_results,
#         include_raw_content=include_raw_content,
#         topic=topic,
#     )


@tool
def basic_web_search(query: str) -> str:
    """Run web search on the question."""
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch
        except ImportError as ex:
            raise ImportError(
                "tavily-python package is required. Install with: uv add tavily-python --group demos"
            ) from ex

        tavily_tool = TavilySearch(max_results=5)
        result = tavily_tool.invoke({"query": query})
        
        # TavilySearch returns a list of Document objects or a string
        # Handle both cases
        if isinstance(result, str):
            import json
            result_dict = json.loads(result)
            docs = result_dict.get("results", [])
            web_results = "\n".join([d["content"] for d in docs])
        elif isinstance(result, list):
            web_results = "\n".join([d["content"] if isinstance(d, dict) else d.page_content for d in result])
        else:
            web_results = str(result)
    else:
        from langchain_community.tools import DuckDuckGoSearchRun

        duckduck_search_tool = DuckDuckGoSearchRun()
        web_results = duckduck_search_tool.invoke(query)

    return web_results
