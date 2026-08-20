"""Every file in results/ must validate against the committed schema. This is
the guarantee the public repo makes."""
import json
from pathlib import Path

import jsonschema
import pytest

RESULTS = Path("results")
SCHEMA = json.loads(Path("schema/result.schema.json").read_text())


@pytest.mark.parametrize(
    "path", sorted(RESULTS.glob("*.json")), ids=lambda p: p.name
)
def test_committed_result_validates_against_schema(path):
    jsonschema.validate(json.loads(path.read_text()), SCHEMA)


def test_schema_file_is_itself_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
