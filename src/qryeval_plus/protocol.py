"""Dataset protocol and corpus audit helpers for the TriviaQA benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from qryeval_plus.config import ConfigError, load_config


EXPECTED_QUESTION_COUNT = 318
EXPECTED_DEV_COUNT = 40
EXPECTED_TEST_COUNT = 278


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: str | Path) -> int:
    candidate = Path(path)
    if candidate.is_file():
        return candidate.stat().st_size
    return sum(item.stat().st_size for item in candidate.rglob("*") if item.is_file())


def read_query_file(path: str | Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ConfigError("Malformed query line {} in {}.".format(number, path))
        qid, question = line.split(":", 1)
        qid, question = qid.strip(), question.strip()
        if not qid or not question:
            raise ConfigError("Malformed query line {} in {}.".format(number, path))
        rows.append((qid, question))
    return rows


def triviaqa_questions(gold_path: str | Path) -> List[Tuple[str, str]]:
    payload = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    rows = [
        (str(item["QuestionId"]), str(item["Question"]).strip())
        for item in payload.get("Data", [])
    ]
    if len(rows) != EXPECTED_QUESTION_COUNT:
        raise ConfigError(
            "Expected {} TriviaQA questions, found {}.".format(
                EXPECTED_QUESTION_COUNT, len(rows)
            )
        )
    if len({qid for qid, _ in rows}) != len(rows):
        raise ConfigError("TriviaQA gold contains duplicate QuestionId values.")
    return rows


def build_split_payload(
    gold_path: str | Path, dev_query_path: str | Path
) -> Dict[str, Any]:
    all_rows = triviaqa_questions(gold_path)
    by_id = dict(all_rows)
    dev_rows = read_query_file(dev_query_path)
    dev_ids = [qid for qid, _ in dev_rows]
    if len(dev_ids) != EXPECTED_DEV_COUNT or len(set(dev_ids)) != EXPECTED_DEV_COUNT:
        raise ConfigError("The frozen development set must contain 40 unique qids.")
    missing = [qid for qid in dev_ids if qid not in by_id]
    if missing:
        raise ConfigError("Development qids missing from gold: {}".format(", ".join(missing)))
    test_rows = [(qid, question) for qid, question in all_rows if qid not in set(dev_ids)]
    if len(test_rows) != EXPECTED_TEST_COUNT:
        raise ConfigError("The frozen test set must contain 278 qids.")
    return {
        "all": all_rows,
        "dev": [(qid, by_id[qid]) for qid in dev_ids],
        "test": test_rows,
    }


def write_split_files(
    gold_path: str | Path,
    dev_query_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Create deterministic query files. Intended for release preparation."""
    payload = build_split_payload(gold_path, dev_query_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in payload.items():
        path = target / "{}.qry".format(name)
        path.write_text(
            "".join("{}: {}\n".format(qid, question) for qid, question in rows),
            encoding="utf-8",
        )
        files[name] = {
            "path": path.name,
            "questions": len(rows),
            "sha256": sha256_file(path),
        }
    manifest = {
        "protocol": "triviaqa-318-v1",
        "development_policy": "The previously analyzed 40 questions are development-only.",
        "test_policy": "The remaining 278 qids are a locked test set.",
        "files": files,
        "gold_sha256": sha256_file(gold_path),
    }
    manifest_path = target / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_split_manifest(
    manifest_path: str | Path,
    gold_path: str | Path,
    qrel_path: str | Path,
) -> Dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    base = manifest_file.parent
    ids: Dict[str, set[str]] = {}
    for name, expected in (("all", 318), ("dev", 40), ("test", 278)):
        record = manifest["files"][name]
        path = Path(record["path"])
        if not path.is_absolute():
            path = base / path
        rows = read_query_file(path)
        if len(rows) != expected or sha256_file(path) != record["sha256"]:
            raise ConfigError("Split file failed validation: {}".format(path))
        ids[name] = {qid for qid, _ in rows}
    if ids["dev"] & ids["test"] or ids["dev"] | ids["test"] != ids["all"]:
        raise ConfigError("Development and test splits are not disjoint and exhaustive.")
    gold_rows = triviaqa_questions(gold_path)
    gold = dict(gold_rows)
    gold_ids = set(gold)
    qrel_ids = {
        line.split()[0]
        for line in Path(qrel_path).read_text(encoding="utf-8").splitlines()
        if line.split()
    }
    missing_gold = ids["all"] - gold_ids
    missing_qrels = ids["all"] - qrel_ids
    if missing_gold or missing_qrels:
        raise ConfigError(
            "Protocol coverage failed: {} missing gold, {} missing qrels.".format(
                len(missing_gold), len(missing_qrels)
            )
        )
    if manifest.get("gold_sha256") != sha256_file(gold_path):
        raise ConfigError("TriviaQA gold checksum does not match the frozen manifest.")
    all_path = Path(manifest["files"]["all"]["path"])
    if not all_path.is_absolute():
        all_path = base / all_path
    if any(gold.get(qid) != question for qid, question in read_query_file(all_path)):
        raise ConfigError("Frozen query text does not match TriviaQA gold.")
    return {"questions": 318, "dev": 40, "test": 278, "validated": True}


def inspect_corpus(config_path: str | Path) -> Dict[str, Any]:
    """Inspect the mounted Lucene/FAISS assets and return auditable metadata."""
    parameters = load_config(config_path)
    index_path = Path(parameters["indexPath"])
    dense_path = _find_parameter(parameters, "dense:indexPath")
    passage = _passage_parameters(parameters)

    from qryeval_plus.core.Idx import Idx

    opened = False
    try:
        opened = bool(Idx.open(str(index_path)))
        if not opened:
            raise RuntimeError("Unable to open index: {}".format(index_path))
        doc_count = int(Idx.getNumDocs())
        fields = {}
        for field in ("url", "keywords", "title", "body", "inlink"):
            count = int(Idx.getDocCount(field))
            total = int(Idx.getSumOfFieldLengths(field))
            fields[field] = {
                "documents": count,
                "terms": total,
                "average_terms": (total / count) if count else 0.0,
            }
    finally:
        if Idx.indexReader is not None:
            Idx.close()

    dense_vectors = None
    dense_bytes = None
    if dense_path:
        import faiss

        dense = Path(dense_path)
        dense_index = faiss.read_index(str(dense))
        dense_vectors = int(dense_index.ntotal)
        dense_bytes = directory_size(dense)
        if dense_vectors != doc_count:
            raise ConfigError(
                "FAISS/Lucene mismatch: {} vectors for {} documents.".format(
                    dense_vectors, doc_count
                )
            )

    lucene_sha = sha256_path(index_path)
    dense_sha = sha256_path(dense_path) if dense_path else None
    version_digest = hashlib.sha256((lucene_sha + (dense_sha or "")).encode("ascii")).hexdigest()[:16]
    return {
        "corpus_version": version_digest,
        "lucene": {
            "path": str(index_path.resolve()),
            "documents": doc_count,
            "bytes": directory_size(index_path),
            "sha256": lucene_sha,
            "fields": fields,
        },
        "faiss": {
            "path": str(Path(dense_path).resolve()) if dense_path else None,
            "vectors": dense_vectors,
            "bytes": dense_bytes,
            "sha256": dense_sha,
        },
        "passages": {
            **passage,
            "storage": "dynamic_per_retrieved_document",
            "precomputed_chunk_count": None,
        },
    }


def _find_parameter(parameters: Dict[str, Any], key: str) -> str | None:
    if parameters.get(key):
        return str(parameters[key])
    for value in parameters.values():
        if isinstance(value, dict) and value.get(key):
            return str(value[key])
    return None


def _passage_parameters(parameters: Dict[str, Any]) -> Dict[str, int]:
    defaults = {"length": 150, "stride": 140, "candidates_per_document": 6}
    for value in parameters.values():
        if isinstance(value, dict) and any(key.startswith("rag:psg") for key in value):
            return {
                "length": int(value.get("rag:psgLen", defaults["length"])),
                "stride": int(value.get("rag:psgStride", defaults["stride"])),
                "candidates_per_document": int(
                    value.get("rag:psgCnt", defaults["candidates_per_document"])
                ),
            }
    return defaults


def sha256_path(path_value: str | Path) -> str:
    """Hash a file or a directory tree by relative name and file content."""
    path = Path(path_value)
    digest = hashlib.sha256()
    entries: Iterable[Path] = [path] if path.is_file() else sorted(path.rglob("*"))
    for entry in entries:
        if not entry.is_file():
            continue
        relative = entry.relative_to(path) if path.is_dir() else Path(entry.name)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
