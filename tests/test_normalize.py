"""Tests for text normalization."""

from er.normalize import normalize


class TestBasicNormalization:
    def test_lowercase(self):
        assert normalize("Microsoft SQL Server") == "microsoft sql server"

    def test_strip_punctuation(self):
        assert normalize("noah's ark (jewel case)") == "noah s ark jewel case"

    def test_collapse_whitespace(self):
        assert normalize("foo   bar    baz") == "foo bar baz"

    def test_strip_leading_trailing(self):
        assert normalize("  hello world  ") == "hello world"


class TestStopTokenRemoval:
    def test_removes_legal_tokens(self):
        assert normalize("Acme Inc Software") == "acme software"

    def test_removes_common_stop_words(self):
        assert normalize("The Best Software for the Enterprise") == "best software enterprise"

    def test_preserves_non_stop_tokens(self):
        assert normalize("adobe after effects") == "adobe after effects"

    def test_case_insensitive_stop_removal(self):
        assert normalize("INC Corp LLC") == ""


class TestDigitSplitting:
    def test_splits_version_number(self):
        assert normalize("v5.0") == "v 5 0"

    def test_splits_model_number(self):
        result = normalize("Pro9000")
        assert "pro" in result
        assert "9000" in result

    def test_preserves_pure_numbers(self):
        assert normalize("2007") == "2007"

    def test_splits_mixed_alphanumeric(self):
        assert normalize("3pk") == "3 pk"


class TestEdgeCases:
    def test_none_input(self):
        assert normalize(None) == ""

    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_punctuation(self):
        assert normalize("---!!!...") == ""

    def test_only_stop_words(self):
        assert normalize("the and of for") == ""

    def test_unicode_preserved(self):
        assert normalize("café résumé") == "café résumé"

    def test_numbers_only(self):
        assert normalize("12345") == "12345"

    def test_single_character(self):
        assert normalize("x") == "x"


class TestRealProductNames:
    """Test with actual product names from the Amazon-Google dataset."""

    def test_software_with_version(self):
        result = normalize("Adobe After Effects Professional 6.5")
        assert result == "adobe after effects professional 6 5"

    def test_product_with_parens(self):
        result = normalize("clickart 950 000 premier image pack ( dvd-rom )")
        assert "clickart" in result
        assert "dvd" in result
        assert "rom" in result

    def test_product_with_slash(self):
        result = normalize("ca international arcserve lap/desktop oem 30pk")
        assert "lap" in result
        assert "desktop" in result
        assert "30" in result
