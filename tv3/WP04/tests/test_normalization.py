from wp04.normalization import normalize_text


def test_normalization_folds_vietnamese_diacritics_and_lowercases():
    value = normalize_text("  Bánh Mì ")
    assert value.canonical == "bánh mì"
    assert value.without_diacritics == "banh mi"
