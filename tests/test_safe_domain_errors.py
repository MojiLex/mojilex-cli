from __future__ import annotations

import json

import pytest
import typer
from pydantic import ValidationError
from pydantic_core import PydanticCustomError
from typer.testing import CliRunner

from mojilex_cli.commands.runtime import execute, structured_exception
from mojilex_cli.config import TelegramConfig
from mojilex_cli.domain import Emoji
from test_dataset_helpers import write_fixture

PRIVATE_INPUT = "private-generated-text-should-never-appear"
PRIVATE_KEY = "private-input-key-should-never-appear"
PRIVATE_MESSAGE = "private-validator-message-should-never-appear"


def _domain_error(tmp_path):
    snapshot = write_fixture(tmp_path / "dataset")
    payload = next(iter(snapshot.emojis.values())).model_dump(mode="json")
    payload["media"][0]["width"] = PRIVATE_INPUT
    payload[PRIVATE_KEY] = PRIVATE_INPUT
    with pytest.raises(ValidationError) as caught:
        Emoji.model_validate(payload)
    return caught.value


def _assert_private_omitted(text):
    for value in (PRIVATE_INPUT, PRIVATE_KEY, PRIVATE_MESSAGE, "input_value", "input_type"):
        assert value not in text


def test_domain_validation_reports_only_schema_owned_paths_and_closed_codes(tmp_path):
    error = structured_exception(_domain_error(tmp_path), debug=True)
    assert error.code == "VALIDATION_FAILED"
    assert "$.media[].width (int_type)" in error.message
    assert "$.<unknown-field> (extra_forbidden)" in error.message
    assert error.details == {"exception_type": "ValidationError"}
    _assert_private_omitted(str(error))


def test_model_validator_message_context_input_and_custom_codes_are_not_echoed():
    error = ValidationError.from_exception_data(
        PRIVATE_MESSAGE,
        [
            {
                "type": "value_error",
                "loc": (),
                "input": {PRIVATE_KEY: PRIVATE_INPUT},
                "ctx": {"error": ValueError(PRIVATE_MESSAGE)},
            },
            {
                "type": PydanticCustomError(PRIVATE_MESSAGE, PRIVATE_MESSAGE),
                "loc": (PRIVATE_KEY,),
                "input": PRIVATE_INPUT,
            },
        ],
    )
    reported = structured_exception(error)
    assert reported.message == (
        "Model validation failed: $ (value_error); $.<unknown-field> (schema_error)"
    )
    _assert_private_omitted(str(reported))


def test_location_unknown_mapping_key_is_not_printed_even_with_known_model_title():
    error = ValidationError.from_exception_data(
        "Emoji",
        [{"type": "int_type", "loc": ("media", 999999, PRIVATE_KEY), "input": PRIVATE_INPUT}],
    )
    reported = structured_exception(error)
    assert reported.message == "Model validation failed: $.media[].<unknown-field> (int_type)"
    assert "999999" not in reported.message
    _assert_private_omitted(str(reported))


def test_config_validation_keeps_configuration_exit_classification():
    with pytest.raises(ValidationError) as caught:
        TelegramConfig(max_attempts=0)
    reported = structured_exception(caught.value)
    assert reported.code == "CONFIG_INVALID"
    assert "$.max_attempts (greater_than_equal)" in reported.message
    assert "input_value" not in reported.message


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_cli_validation_error_has_no_payload_in_human_json_or_debug(tmp_path, json_output, debug):
    error = _domain_error(tmp_path)
    application = typer.Typer()

    def fail():
        raise error from ValueError(PRIVATE_MESSAGE)

    @application.command()
    def check():
        execute("describe", fail, json_output=json_output, debug=debug)

    result = CliRunner().invoke(application, [])
    assert result.exit_code == 10, result.output
    _assert_private_omitted(result.output)
    assert "VALIDATION_FAILED" in result.output
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["errors"][0]["code"] == "VALIDATION_FAILED"
        assert result.stderr == ""
    elif debug:
        assert "Traceback (validation input omitted)" in result.output
        assert "test_safe_domain_errors.py" in result.output


def test_validation_summary_is_bounded_for_many_errors():
    error = ValidationError.from_exception_data(
        "Emoji",
        [
            {"type": "extra_forbidden", "loc": (f"{PRIVATE_KEY}-{i}",), "input": PRIVATE_INPUT}
            for i in range(100)
        ],
    )
    reported = structured_exception(error)
    assert "additional errors omitted" in reported.message
    assert len(reported.message) < 200
    _assert_private_omitted(str(reported))
