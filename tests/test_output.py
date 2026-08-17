import json

from qryeval_plus.io.Output import Output


def test_answer_output_creates_parent_directories(tmp_path):
    output_path = tmp_path / "deep" / "answers.json"
    prompt_path = tmp_path / "deep" / "prompts.txt"
    metadata_path = tmp_path / "deep" / "llm.json"
    batch = {"1": {
        "answer": "example",
        "prompt_rag": [{"role": "user", "content": "example"}],
        "llm": {"provider": "mock", "success": True},
    }}

    Output({
        "type": "triviaqa_evaluation",
        "outputPath": str(output_path),
        "promptPath": str(prompt_path),
        "metadataPath": str(metadata_path),
    }).execute(batch)

    assert json.loads(output_path.read_text()) == {"1": "example"}
    assert prompt_path.is_file()
    assert json.loads(metadata_path.read_text()) == {
        "1": {"provider": "mock", "success": True}
    }
