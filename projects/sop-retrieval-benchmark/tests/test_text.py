import pytest

from sop_retrieval_benchmark.text import tokenize


def test_lowercases_and_drops_stopwords():
    assert tokenize("The Return window is 30 days") == ["return", "window", "30", "day"]


def test_site_codes_and_decimals_stay_whole():
    assert tokenize("Site BOM-02 holds 2.5 litres") == ["site", "bom-02", "hold", "2.5", "litre"]


def test_plural_and_ing_forms_match():
    assert tokenize("pallets") == tokenize("pallet")
    assert tokenize("loading") == tokenize("load")


def test_words_ending_in_ss_keep_their_s():
    # Singular, plural and -ing forms must reduce to the same token (#13).
    assert len(set(tokenize("process processes processing"))) == 1
    assert tokenize("glass") == tokenize("glasses") == ["glass"]
    assert tokenize("access less") == ["access", "less"]


def test_short_words_not_stemmed():
    assert tokenize("gas bus") == ["gas", "bus"]
    assert tokenize("bays") == ["bay"]


def test_rejects_non_string():
    with pytest.raises(TypeError):
        tokenize(None)  # type: ignore[arg-type]
