"""Unit tests for language detection and stopwords utilities in genai_tk.extra.nlp."""

from genai_tk.extra.nlp.language import (
    BUSINESS_LANGUAGES,
    detect_language,
    get_language_detector,
)
from genai_tk.extra.nlp.stopwords import (
    get_dominant_language,
    get_ladybug_stemmer,
    get_stopwords,
    get_stopwords_union,
    stem_stopwords,
)


class TestLanguageDetection:
    """Tests for detect_language using lingua-py."""

    def test_detect_english(self):
        text = "This is a comprehensive quarterly financial report detailing revenue, expenses, and growth projections."
        assert detect_language(text) == "en"

    def test_detect_french(self):
        text = (
            "Ce document présente le rapport financier annuel ainsi que les perspectives de croissance de l'entreprise."
        )
        assert detect_language(text) == "fr"

    def test_detect_german(self):
        text = "Dieser Geschäftsbericht enthält detaillierte Informationen über Umsatz, Gewinn und zukünftige Investitionen."
        assert detect_language(text) == "de"

    def test_detect_spanish(self):
        text = "Este informe financiero describe los resultados del ejercicio fiscal y el plan estratégico para el próximo año."
        assert detect_language(text) == "es"

    def test_detect_italian(self):
        text = "La presente relazione di bilancio illustra l'andamento economico e finanziario della società."
        assert detect_language(text) == "it"

    def test_detect_chinese(self):
        text = "本财务年度报告详细总结了公司的运营状况、营业收入、利润分配及未来的战略发展规划。"
        assert detect_language(text) == "zh"

    def test_detect_japanese(self):
        text = "本報告書は当社の年度決算概要および今後の事業戦略について説明するものです。"
        assert detect_language(text) == "ja"

    def test_detect_short_or_empty_text(self):
        assert detect_language("") is None
        assert detect_language("   ") is None
        assert detect_language("Hello") is None  # Below default min_length=15

    def test_custom_language_filter(self):
        text = "Ce document est rédigé en français."
        # Restrict to only en and fr
        assert detect_language(text, languages=["en", "fr"]) == "fr"

    def test_cached_detector_instance(self):
        d1 = get_language_detector()
        d2 = get_language_detector()
        assert d1 is d2
        assert len(BUSINESS_LANGUAGES) > 10


class TestStopwordsAndStemmers:
    """Tests for stop words and Ladybug stemmer mapping."""

    def test_get_stopwords_english(self):
        stops = get_stopwords("en")
        assert "the" in stops
        assert "and" in stops
        assert "is" in stops
        assert len(stops) > 50

    def test_get_stopwords_french(self):
        stops = get_stopwords("fr")
        assert "le" in stops or "les" in stops
        assert "de" in stops
        assert len(stops) > 50

    def test_get_stopwords_german(self):
        stops = get_stopwords("de")
        assert "und" in stops
        assert "der" in stops or "die" in stops
        assert len(stops) > 50

    def test_get_stopwords_union(self):
        union_stops = get_stopwords_union(["en", "fr"])
        assert "the" in union_stops
        assert "les" in union_stops or "le" in union_stops
        assert len(union_stops) > len(get_stopwords("en"))

    def test_ladybug_stemmer_mapping(self):
        assert get_ladybug_stemmer("en") == "english"
        assert get_ladybug_stemmer("fr") == "french"
        assert get_ladybug_stemmer("de") == "german"
        assert get_ladybug_stemmer("es") == "spanish"
        assert get_ladybug_stemmer("it") == "italian"
        assert get_ladybug_stemmer("nl") == "dutch"
        assert get_ladybug_stemmer("no") == "norwegian"
        assert get_ladybug_stemmer("nb") == "norwegian"
        assert get_ladybug_stemmer("zh") == "none"
        assert get_ladybug_stemmer("ja") == "none"
        assert get_ladybug_stemmer("unknown_code") == "english"

    def test_dominant_language(self):
        assert get_dominant_language(["fr", "fr", "en", "de"]) == "fr"
        assert get_dominant_language(["en", "en", "fr"]) == "en"
        assert get_dominant_language([]) == "en"
        assert get_dominant_language([], default="fr") == "fr"

    def test_stem_stopwords_english(self):
        stops = get_stopwords("en")
        stemmed = stem_stopwords(stops, "english")
        assert "the" in stemmed
        assert "have" in stemmed  # 'having' stems to 'have'
        assert "having" not in stemmed
        assert len(stemmed) < len(stops)  # stemming collapses inflected forms

    def test_stem_stopwords_french(self):
        stops = get_stopwords("fr")
        stemmed = stem_stopwords(stops, "french")
        assert "le" in stemmed  # 'les' stems to 'le'
        assert "lès" not in stemmed  # 'lès' stems to 'les'
        assert len(stemmed) < len(stops)  # stemming collapses inflected forms

    def test_stem_stopwords_none_stemmer_returns_input(self):
        raw = {"having", "les"}
        assert stem_stopwords(raw, "none") == raw

    def test_stem_stopwords_unknown_stemmer_returns_input(self):
        raw = {"having"}
        assert stem_stopwords(raw, "not_a_snowball_language") == raw
