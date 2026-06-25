"""Tests for the SpaCy model manager utility."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from genai_tk.extra.nlp.model_manager import SpaCyModelManager


class TestSpaCyModelManager:
    """Test cases for SpaCyModelManager class."""

    def test_get_model_path(self, tmp_path: Path) -> None:
        """Test that model path is correctly constructed."""
        with patch("genai_tk.extra.nlp.model_manager.global_config") as mock_config:
            mock_config.return_value.get_dir_path.return_value = Path(tmp_path)

            model_name = "en_core_web_sm"
            expected_path = Path(tmp_path) / "spacy_models" / model_name

            result = SpaCyModelManager.get_model_path(model_name)

            assert result == expected_path
            mock_config.return_value.get_dir_path.assert_called_once_with("paths.models", create_if_not_exists=True)

    def test_is_model_installed_true(self, tmp_path: Path) -> None:
        """Test is_model_installed returns True when model exists."""
        model_name = "en_core_web_sm"
        model_path = tmp_path / "spacy_models" / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        with patch("genai_tk.extra.nlp.model_manager.SpaCyModelManager.get_model_path", return_value=Path(model_path)):
            assert SpaCyModelManager.is_model_installed(model_name) is True

    def test_is_model_installed_false(self, tmp_path: Path) -> None:
        """Test is_model_installed returns False when model doesn't exist."""
        model_name = "en_core_web_sm_nonexistent"  # use a name that is definitely not installed
        model_path = tmp_path / "spacy_models" / model_name

        with patch("genai_tk.extra.nlp.model_manager.SpaCyModelManager.get_model_path", return_value=Path(model_path)):
            assert SpaCyModelManager.is_model_installed(model_name) is False

    @patch("subprocess.run")
    def test_download_model_not_installed(self, mock_subprocess: MagicMock, tmp_path: Path) -> None:
        """Test download_model when model is not installed."""
        model_name = "en_core_web_sm"
        model_path = tmp_path / "spacy_models" / model_name

        with patch("genai_tk.extra.nlp.model_manager.SpaCyModelManager.get_model_path", return_value=Path(model_path)):
            with patch("genai_tk.extra.nlp.model_manager.SpaCyModelManager.is_model_installed", return_value=False):
                result = SpaCyModelManager.download_model(model_name)

                assert result == Path(model_path)
                mock_subprocess.assert_called_once_with(
                    ["python", "-m", "spacy", "download", model_name],
                    check=True,
                )

    @patch("subprocess.run")
    def test_download_model_already_installed(self, mock_subprocess: MagicMock, tmp_path: Path) -> None:
        """Test download_model when model is already installed."""
        model_name = "en_core_web_sm"
        model_path = tmp_path / "spacy_models" / model_name
        model_path.mkdir(parents=True, exist_ok=True)  # Simulate already-present model dir

        with patch("genai_tk.extra.nlp.model_manager.SpaCyModelManager.get_model_path", return_value=Path(model_path)):
            result = SpaCyModelManager.download_model(model_name)

            assert result == Path(model_path)
            mock_subprocess.assert_not_called()

    @patch("genai_tk.extra.nlp.model_manager.SpaCyModelManager.download_model")
    def test_setup_spacy_model_not_installed(self, mock_download: MagicMock, tmp_path: Path) -> None:
        """Test setup_spacy_model when model is not installed."""
        model_name = "en_core_web_sm"
        model_path = tmp_path / "spacy_models" / model_name

        with patch("genai_tk.extra.nlp.model_manager.SpaCyModelManager.get_model_path", return_value=Path(model_path)):
            with patch("builtins.__import__") as mock_import:
                mock_spacy = MagicMock()
                mock_spacy.load.side_effect = [OSError, None]  # First call fails, second succeeds
                mock_import.return_value = mock_spacy
                SpaCyModelManager.setup_spacy_model(model_name)
                mock_download.assert_called_once()

    @patch("genai_tk.extra.nlp.model_manager.SpaCyModelManager.download_model")
    def test_setup_spacy_model_already_installed(self, mock_download: MagicMock, tmp_path: Path) -> None:
        """Test setup_spacy_model when model is already installed."""
        model_name = "en_core_web_sm"
        model_path = tmp_path / "spacy_models" / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        with patch("genai_tk.extra.nlp.model_manager.SpaCyModelManager.get_model_path", return_value=Path(model_path)):
            with patch("builtins.__import__") as mock_import:
                mock_spacy = MagicMock()
                mock_spacy.load.return_value = None  # Model loads successfully
                mock_import.return_value = mock_spacy
                SpaCyModelManager.setup_spacy_model(model_name)
                mock_download.assert_not_called()
