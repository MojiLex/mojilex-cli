import json

from mojilex_cli.output import ExitCode, OutputEnvelope, RunStatus, StructuredError


def test_machine_output_is_one_compact_secret_free_object() -> None:
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
    envelope = OutputEnvelope(
        ok=False,
        command="add",
        status=RunStatus.FAILED,
        run_id="01TEST",
        errors=[
            StructuredError(
                code="AUTH_FAILED",
                message=f"credential {token} rejected",
                retryable=False,
                hint="Replace the token.",
                details={"authorization": token},
            )
        ],
    )
    encoded = envelope.to_json()
    assert "\n" not in encoded
    assert token not in encoded
    assert json.loads(encoded)["errors"][0]["details"]["authorization"] == "[REDACTED]"
    assert envelope.exit_code is ExitCode.AUTH


def test_partial_result_has_partial_exit_code() -> None:
    envelope = OutputEnvelope(
        ok=False,
        command="add",
        status=RunStatus.PARTIAL,
        run_id="01TEST",
        errors=[
            StructuredError(
                code="MEDIA_INVALID",
                message="One collection failed.",
                retryable=False,
                hint="Inspect that collection.",
            )
        ],
    )
    assert envelope.exit_code is ExitCode.PARTIAL


def test_redaction_covers_named_credentials_headers_and_urls() -> None:
    secrets = {
        "access": "opaque-access-value",
        "client": "opaque-client-value",
        "bearer": "opaque-bearer-value",
        "basic": "b3BhcXVlLWJhc2ljLXZhbHVl",
        "username": "credential-user",
        "password": "credential-password",
        "query": "opaque-query-value",
    }
    envelope = OutputEnvelope(
        ok=True,
        command="doctor",
        status=RunStatus.SUCCEEDED,
        run_id="01TEST",
        result={
            "access_token": secrets["access"],
            "nested": {
                "client_secret": secrets["client"],
                "authorization_header": f"Bearer {secrets['bearer']}",
            },
            "message": (
                f"Authorization: Basic {secrets['basic']}; fallback Bearer {secrets['bearer']}"
            ),
            "url": (
                f"https://{secrets['username']}:{secrets['password']}@example.test/path"
                f"?access_token={secrets['query']}"
            ),
        },
    )

    encoded = envelope.to_json()
    for secret in secrets.values():
        assert secret not in encoded
    assert encoded.count("[REDACTED]") >= 6
