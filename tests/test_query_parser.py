from qryeval_plus.query.QryParser import QryParser
from qryeval_plus.retrieval.Ranker import _prepare_bow_query


def test_natural_language_period_is_not_treated_as_unknown_field():
    query = QryParser.getQuery(
        "#sum(It became known as the Chile Pine. What is its current name?)"
    )
    assert query is not None
    assert "chile" in str(query).lower()


def test_explicit_supported_field_suffix_is_preserved():
    query = QryParser.getQuery("#sum(monkey.title)")
    assert query is not None
    assert ".title" in str(query).lower()


def test_abbreviation_with_periods_is_analyzed_as_natural_text():
    query = QryParser.getQuery("#sum(Which U.S. agency was involved?)")
    assert query is not None


def test_bow_preparation_removes_unbalanced_natural_language_parenthesis():
    prepared = _prepare_bow_query(
        "Who directed 'Last Tango in Paris'? (", "#sum"
    )
    assert prepared.count("(") == prepared.count(")") == 1
    assert "last tango" in prepared


def test_bow_preparation_preserves_explicit_prf_structure():
    structured = "#WSUM(0.5 #SUM(original) 0.5 #WSUM(1 term.body))"
    prepared = _prepare_bow_query(structured, "#sum")
    assert prepared == "#sum({})".format(structured)
