"""Contract tests for the Sonar workflow's dependabot handling (issue #59).

Dependabot-triggered ``pull_request`` events never receive repo secrets
(GitHub platform behavior), so the pull_request Sonar scan 401s against the
runner-local SonarQube on every dependabot PR. The contract under test:

- the pull_request job defers dependabot PRs with a visible notice instead of
  failing, and none of its scan steps run for them;
- a workflow_run job (which does receive secrets) performs the real scan,
  restricted to this repository's dependabot PRs with a passing CI run;
- the workflow_run trigger never reaches the push/pull_request job (no
  redundant default-branch scans cancelling push-triggered runs);
- PR-number resolution fails loudly on API errors (no silent green bypass)
  and only skips when the API confirms no pull request exists.
"""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "sonar.yml"


def _load_workflow() -> dict:
    with WORKFLOW.open() as fh:
        return yaml.safe_load(fh)


def _triggers(workflow: dict) -> dict:
    triggers = workflow.get("on") or workflow.get(True)
    assert isinstance(triggers, dict)
    return triggers


def _dependabot_job(jobs: dict) -> dict:
    gated = [j for j in jobs.values() if "workflow_run" in str(j.get("if", ""))]
    assert gated, "sonar.yml must have a workflow_run-gated job for dependabot PRs"
    assert len(gated) == 1, "expected exactly one workflow_run-gated Sonar job"
    return gated[0]


def _step_with_id(job: dict, step_id: str) -> dict:
    steps = [s for s in job["steps"] if s.get("id") == step_id]
    assert steps, f"job must have a step with id '{step_id}'"
    return steps[0]


def test_workflow_run_triggers_on_completed_ci() -> None:
    workflow_run = _triggers(_load_workflow())["workflow_run"]
    assert "CI" in workflow_run["workflows"]
    assert "completed" in workflow_run["types"]


def test_pull_request_job_never_runs_on_workflow_run_events() -> None:
    job = _load_workflow()["jobs"]["sonarqube"]
    job_if = str(job["if"])
    assert "github.event_name == 'push'" in job_if
    assert "github.event_name == 'pull_request'" in job_if
    assert "!= 'pull_request'" not in job_if


def test_dependabot_job_is_restricted_to_this_repo() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    job_if = str(job["if"])
    assert "github.event_name == 'workflow_run'" in job_if
    assert "github.event.workflow_run.event == 'pull_request'" in job_if
    assert "github.event.workflow_run.conclusion == 'success'" in job_if
    assert "github.event.workflow_run.head_repository.full_name" in job_if
    assert "github.repository" in job_if


def test_dependabot_job_gates_on_exact_bot_identity_and_branch() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    job_if = str(job["if"])
    assert "== 'dependabot[bot]'" in job_if
    assert "startsWith(github.event.workflow_run.head_branch, 'dependabot/')" in job_if
    assert "contains(" not in job_if, "substring actor matching is too broad"


def test_dependabot_job_scans_with_secrets_and_merge_ref() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    scan = _step_with_id(job, "scan")
    assert "secrets.SONAR_TOKEN" in str(scan.get("env", {}).get("SONAR_TOKEN", ""))
    checkout = next(s for s in job["steps"] if "actions/checkout" in str(s.get("uses", "")))
    assert "refs/pull/" in str(checkout.get("with", {}).get("ref", ""))


def test_dependabot_job_can_read_pull_requests() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    permissions = job["permissions"]
    assert permissions["contents"] == "read"
    assert permissions["pull-requests"] == "read"


def test_dependabot_job_resolves_pr_with_jq_and_fails_loudly() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    resolve = _step_with_id(job, "pr")
    run = str(resolve["run"])
    assert "jq -r '.[0].number // empty'" in run
    assert "curl -fsS" in run
    assert "|| true" not in run, "API errors must fail the job, not silently pass it"
    assert "grep" not in run, "parse API JSON with jq, not grep"


def test_pull_request_job_defers_dependabot_with_notice() -> None:
    job = _load_workflow()["jobs"]["sonarqube"]
    preflight = _step_with_id(job, "preflight")
    assert "github.event_name == 'pull_request'" in str(preflight.get("if", ""))
    run = str(preflight["run"])
    assert '"$PR_AUTHOR" = "dependabot[bot]"' in run
    assert "::notice::" in run
    assert 'echo "deferred=true" >> "$GITHUB_OUTPUT"' in run
    assert "*dependabot*" not in run, "substring author matching is too broad"


def test_pull_request_job_skips_scan_steps_when_deferred() -> None:
    job = _load_workflow()["jobs"]["sonarqube"]
    for step_id in ("scan", "enforce"):
        step = _step_with_id(job, step_id)
        assert "steps.preflight.outputs.deferred != 'true'" in str(step.get("if", ""))


def test_workflow_uses_loopback_ip_not_localhost() -> None:
    text = WORKFLOW.read_text()
    assert "localhost:9001" not in text
