"""Replacements for deprecated langchain-community functionality.

This module provides standalone implementations of commonly-used langchain-community
components that are marked for sunset. They are collected here to make it easy to
switch back to langchain-community or migrate to alternative solutions if needed.

Modules:
- fastembed_embeddings: FastEmbed-backed LangChain embeddings (replaces langchain_community.embeddings.fastembed.FastEmbedEmbeddings)
- sqlite_cache: Stdlib SQLite-backed LLM cache (replaces langchain_community.cache.SQLiteCache)
"""
