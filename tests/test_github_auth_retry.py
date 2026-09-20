import pytest

from mojilex_cli.github import GitHubCLI, GitHubError


@pytest.mark.parametrize(
    "failure",
    [
        "Timeout trying to log in to github.com account test (keyring)",
        "GitHub CLI request exceeded its time limit",
        "dial tcp: i/o timeout",
        "net/http: TLS handshake timeout",
        "connection reset by peer",
    ],
)
def test_auth_recovers_from_temporary_transport_failure(monkeypatch, failure):
    calls = []
    sleeps = []

    def run(self, *args):
        calls.append(args)
        if len(calls) < 3:
            raise GitHubError(failure)
        return "authenticated"

    monkeypatch.setattr(GitHubCLI, "run", run)
    monkeypatch.setattr("mojilex_cli.github.client.time.sleep", sleeps.append)
    GitHubCLI().auth_status()
    assert calls == [("auth", "status", "--hostname", "github.com")] * 3
    assert sleeps == [1, 2]


def test_auth_retries_are_bounded(monkeypatch):
    calls = []
    sleeps = []

    def run(self, *args):
        calls.append(args)
        raise GitHubError("Timeout trying to log in to github.com")

    monkeypatch.setattr(GitHubCLI, "run", run)
    monkeypatch.setattr("mojilex_cli.github.client.time.sleep", sleeps.append)
    with pytest.raises(GitHubError, match="Timeout"):
        GitHubCLI().auth_status()
    assert len(calls) == 3 and sleeps == [1, 2]


@pytest.mark.parametrize(
    "failure",
    [
        "Bad credentials (HTTP 401)",
        "HTTP 403",
        "invalid token; Timeout trying to log in to another account",
        "GitHub CLI executable was not found",
        "not logged into any GitHub hosts",
    ],
)
def test_auth_does_not_retry_permanent_failures(monkeypatch, failure):
    calls = []

    def run(self, *args):
        calls.append(args)
        raise GitHubError(failure)

    def unexpected_sleep(*args):
        pytest.fail("permanent failures must not be retried")

    monkeypatch.setattr(GitHubCLI, "run", run)
    monkeypatch.setattr("mojilex_cli.github.client.time.sleep", unexpected_sleep)
    with pytest.raises(GitHubError):
        GitHubCLI().auth_status()
    assert len(calls) == 1


def test_publication_mutation_is_not_retried(monkeypatch):
    calls = []

    def execute(self, *args):
        calls.append(args)
        return "", "Timeout trying to log in to github.com", 1

    monkeypatch.setattr(GitHubCLI, "_execute", execute)
    with pytest.raises(GitHubError):
        GitHubCLI().run("pr", "create")
    assert calls == [("pr", "create")]
