from qryeval_plus.grounding import grounding_status


def test_grounding_status_handles_supported_unsupported_and_empty_answers():
    evidence = ["The V-2 rocket was the first human-made object to enter space."]
    assert grounding_status("V-2 rocket", evidence) == "answered"
    assert grounding_status("Saturn V", evidence) == "needs_review"
    assert grounding_status("", evidence) == "abstained"
    assert grounding_status("answer", []) == "abstained"
