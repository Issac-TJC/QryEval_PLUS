from qryeval_plus.core.Idx import Idx
from qryeval_plus.query.QryParser import QryParser
from qryeval_plus.rewrite.RewriteWithPrf import RewriteWithPrf


def test_prf_reconstructs_terms_from_stored_field_when_vectors_are_absent(monkeypatch):
    class EmptyVector:
        def stemsLength(self):
            return 0

        def positionsLength(self):
            return 0

    monkeypatch.setattr(Idx, "getTermVector", lambda docid, field: EmptyVector())
    monkeypatch.setattr(
        Idx, "getAttribute",
        lambda name, docid: "Potato beetle beetle larvae" if name == "body-string" else None,
    )
    monkeypatch.setattr(
        QryParser, "tokenizeString", lambda value: value.lower().split()
    )
    RewriteWithPrf._cached_document_terms.cache_clear()

    terms, length, source = RewriteWithPrf._cached_document_terms(7, "body")

    assert terms == {"potato": 1, "beetle": 2, "larvae": 1}
    assert length == 4
    assert source == "stored_field"
    RewriteWithPrf._cached_document_terms.cache_clear()


def test_prf_rejects_empty_feedback_instead_of_silently_using_original_query(monkeypatch):
    rewriter = RewriteWithPrf({"prf:algorithm": "rm3"})
    monkeypatch.setattr(
        RewriteWithPrf, "_cached_document_terms",
        staticmethod(lambda docid, field: ({}, 0, "stored_field")),
    )

    try:
        rewriter._score_terms([1], [1.0])
    except RuntimeError as exc:
        assert "found no terms" in str(exc)
    else:
        raise AssertionError("empty feedback must fail closed")


def test_prf_tokenizes_natural_language_before_nesting_structured_query(monkeypatch):
    rewriter = RewriteWithPrf(
        {
            "prf:algorithm": "rm3",
            "prf:numTerms": 1,
            "prf:rm3:origWeight": 0.5,
        }
    )
    monkeypatch.setattr(
        rewriter,
        "_score_terms",
        lambda internal_docids, doc_scores: {"paris": 0.25},
    )
    monkeypatch.setattr(Idx, "getInternalDocid", lambda external_id: 1)

    batch = {
        "sfq_1957": {
            "qstring": "Who directed the film 'Last Tango in Paris'? (",
            "ranking": [(1.0, "doc-1")],
        }
    }
    rewritten = rewriter.rewrite(batch)["sfq_1957"]["qstring"]

    parsed = QryParser.getQuery(rewritten)
    assert parsed is not None
    assert "(" not in rewritten.split("#SUM( ", 1)[1].split(" )", 1)[0]
    assert "who.body" in rewritten
    assert "paris.body" in rewritten
