"""Unit tests for genai_tk.agents.langchain.setup."""

from unittest.mock import patch

from genai_tk.agents.langchain.setup import setup_langchain


class TestSetupLangchain:
    def test_returns_true_with_no_llm(self) -> None:
        result = setup_langchain(llm=None)
        assert result is True

    def test_returns_true_with_valid_fake_llm(self) -> None:
        result = setup_langchain(llm="parrot_local@fake")
        assert result is True

    def test_returns_false_with_invalid_llm(self) -> None:
        result = setup_langchain(llm="nonexistent_model@nowhere")
        assert result is False

    def test_sets_debug_mode(self) -> None:
        with patch("genai_tk.agents.langchain.setup.set_debug") as mock_debug:
            setup_langchain(llm=None, lc_debug=True)
            mock_debug.assert_called_once_with(True)

    def test_sets_verbose_mode(self) -> None:
        with patch("genai_tk.agents.langchain.setup.set_verbose") as mock_verbose:
            setup_langchain(llm=None, lc_verbose=True)
            mock_verbose.assert_called_once_with(True)

    def test_does_not_set_debug_when_false(self) -> None:
        with patch("genai_tk.agents.langchain.setup.set_debug") as mock_debug:
            setup_langchain(llm=None, lc_debug=False)
            mock_debug.assert_not_called()

    def test_sets_cache_method(self) -> None:
        with patch("genai_tk.agents.langchain.setup.LlmCache.set_method") as mock_cache:
            setup_langchain(llm=None, cache="memory")
            mock_cache.assert_called_once_with("memory")

    def test_no_cache_not_set_when_none(self) -> None:
        with patch("genai_tk.agents.langchain.setup.LlmCache.set_method") as mock_cache:
            setup_langchain(llm=None, cache=None)
            mock_cache.assert_not_called()

    def test_updates_global_config_with_valid_llm(self) -> None:
        from genai_tk.utils.config_mngr import global_config

        setup_langchain(llm="parrot_local@fake")
        default = global_config().get_str("llm.models.default")
        assert default == "parrot_local@fake"

    def test_all_options_combined(self) -> None:
        with (
            patch("genai_tk.agents.langchain.setup.set_debug") as mock_debug,
            patch("genai_tk.agents.langchain.setup.set_verbose") as mock_verbose,
            patch("genai_tk.agents.langchain.setup.LlmCache.set_method") as mock_cache,
        ):
            result = setup_langchain(llm=None, lc_debug=True, lc_verbose=True, cache="sqlite")
            assert result is True
            mock_debug.assert_called_once_with(True)
            mock_verbose.assert_called_once_with(True)
            mock_cache.assert_called_once_with("sqlite")
