import json

from conftest import PROJECT_ROOT
from qryeval_plus.protocol import build_split_payload, validate_split_manifest


def test_frozen_triviaqa_protocol_is_disjoint_complete_and_covered():
    data = PROJECT_ROOT / "data" / "evaluation" / "triviaqa"
    protocol = PROJECT_ROOT / "datasets" / "triviaqa" / "protocol"
    payload = build_split_payload(
        data / "verified-wikipedia-dev.json",
        PROJECT_ROOT / "datasets" / "triviaqa" / "verified_wikipedia_dev.qry",
    )
    assert len(payload["all"]) == 318
    assert len(payload["dev"]) == 40
    assert len(payload["test"]) == 278
    assert {qid for qid, _ in payload["dev"]}.isdisjoint(
        {qid for qid, _ in payload["test"]}
    )
    result = validate_split_manifest(
        protocol / "split_manifest.json",
        data / "verified-wikipedia-dev.json",
        data / "verified-wikipedia-dev.qrel",
    )
    assert result == {"questions": 318, "dev": 40, "test": 278, "validated": True}


def test_split_manifest_contains_only_relative_release_paths():
    path = PROJECT_ROOT / "datasets" / "triviaqa" / "protocol" / "split_manifest.json"
    payload = json.loads(path.read_text())
    assert {item["questions"] for item in payload["files"].values()} == {40, 278, 318}
    assert all(not item["path"].startswith("/") for item in payload["files"].values())
