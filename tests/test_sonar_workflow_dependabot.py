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

The shared scan steps (config reset, jq, wait, scan, diagnostics) live
in the composite action ``.github/actions/sonar-scan``; both jobs check
out sources caller-side before invoking it (local composite actions
are loaded from the workspace and cannot check out themselves). The
scanner itself enforces the quality gate via
``-Dsonar.qualitygate.wait=true``; a separate post-scan gate check was
removed as dead code (it could only fail on query flakiness, turning
infra blips into red builds).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "sonar.yml"
COMPOSITE = (
    Path(__file__).resolve().parents[1] / ".github" / "actions" / "sonar-scan" / "action.yml"
)


def _load_workflow() -> dict:
    with WORKFLOW.open() as fh:
        return yaml.safe_load(fh)


def _load_composite() -> dict:
    with COMPOSITE.open() as fh:
        return yaml.safe_load(fh)


def _composite_call(job: dict) -> dict:
    calls = [
        s for s in job["steps"]
        if "/.github/actions/sonar-scan" in str(s.get("uses", ""))
    ]
    assert calls, "job must invoke the sonar-scan composite action"
    assert len(calls) == 1, "expected exactly one sonar-scan composite call per job"
    return calls[0]


def test_pull_request_job_invokes_workspace_composite() -> None:
    job = _load_workflow()["jobs"]["sonarqube"]
    uses = str(_composite_call(job).get("uses", ""))
    assert uses == "./.github/actions/sonar-scan", (
        "the pull_request job is restricted to same-repo PRs and must run "
        "the PR's own composite code so changes to it are tested by CI"
    )


def test_dependabot_job_restores_scan_action_from_trusted_main() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    restore = next(
        (
            s
            for s in job["steps"]
            if s.get("name") == "Restore scan action from trusted main"
        ),
        None,
    )
    assert restore is not None, (
        "the dependabot job scans a PR tree in the workspace: without a "
        "restore step, a workspace-relative composite executes PR-sourced "
        "action code with repo secrets"
    )
    assert "steps.pr.outputs.available == 'true'" in str(restore.get("if", ""))
    assert "git checkout origin/main -- .github/actions/sonar-scan" in str(restore["run"])
    call = _composite_call(job)
    assert str(call.get("uses", "")) == "./.github/actions/sonar-scan"
    assert job["steps"].index(restore) < job["steps"].index(call)


def test_restore_removes_pr_added_action_files() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    restore = next(
        s for s in job["steps"] if s.get("name") == "Restore scan action from trusted main"
    )
    run = str(restore["run"])
    assert "rm -rf .github/actions/sonar-scan" in run, (
        "git checkout origin/main only overlays files that exist on main: "
        "files the PR added under the directory survive the restore and sit "
        "next to the trusted action.yml, shadowing any future relative helper"
    )
    assert run.index("rm -rf .github/actions/sonar-scan") < run.index(
        "git checkout origin/main -- .github/actions/sonar-scan"
    )


def test_restore_refuses_symlinked_action_paths() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    restore = next(
        s for s in job["steps"] if s.get("name") == "Restore scan action from trusted main"
    )
    run = str(restore["run"])
    for path in (".github", ".github/actions", ".github/actions/sonar-scan"):
        assert f"[ -L {path} ]" in run, (
            f"rm -rf and git checkout resolve through a planted symlink at "
            f"{path} and can write outside the workspace on the self-hosted "
            "runner"
        )
    assert run.index("[ -L .github ]") < run.index("rm -rf .github/actions/sonar-scan")


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


def _step_with_name(job: dict, name: str) -> dict:
    steps = [s for s in job["steps"] if s.get("name") == name]
    assert steps, f"job must have a step named '{name}'"
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


def test_dependabot_job_gates_on_branch_prefix_only() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    job_if = str(job["if"])
    assert "startsWith(github.event.workflow_run.head_branch, 'dependabot/')" in job_if
    assert "triggering_actor" not in job_if, (
        "run-actor identity diverges from PR author: a human pushing to a "
        "dependabot branch would silently skip the scan"
    )
    assert "contains(" not in job_if, "substring actor matching is too broad"


def test_dependabot_job_scans_with_secrets_and_merge_ref() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    call = _composite_call(job)
    with_ = call.get("with", {})
    assert "secrets.SONAR_TOKEN" in str(with_.get("token", "")), (
        "the dependabot scan must run with the repo secret; the token "
        "reaches the composite via the token input"
    )
    checkout = next(s for s in job["steps"] if "actions/checkout" in str(s.get("uses", "")))
    ref = str(checkout.get("with", {}).get("ref", ""))
    assert "github.event.workflow_run.head_sha" in ref, (
        "checkout must use the CI-validated head SHA: the live merge ref can "
        "advance between CI completion and the scan (TOCTOU)"
    )
    assert "refs/pull/" not in ref


def test_scan_and_gate_target_the_same_project() -> None:
    scan = _step_with_name(_load_composite()["runs"], "SonarQube Scan")
    scanner_opts = str(scan.get("env", {}).get("SONAR_SCANNER_OPTS", ""))
    assert "-Dsonar.projectKey=${{ inputs.project-key }}" in scanner_opts, (
        "the scan must be pinned to the same project-key input the gate "
        "queries: otherwise a PR-controlled sonar-project.properties can "
        "steer the scan to a different project than the gate evaluates"
    )
    assert "-Dsonar.qualitygate.wait=true" in scanner_opts, (
        "the gate-wait flag must not live in the overridable scanner-opts "
        "input: a caller override could otherwise drop it and race the "
        "separate gate check with an UNKNOWN result"
    )


def test_install_jq_fails_loudly_without_homebrew() -> None:
    step = next(s for s in _load_composite()["runs"]["steps"] if s.get("name") == "Install jq")
    run = str(step["run"])
    assert "command -v brew" in run and "::error::" in run, (
        "brew is the only install path: on a runner without Homebrew the "
        "step must fail with an explicit error, not a cryptic command-not-found"
    )
    description = str(_load_composite().get("description", ""))
    assert "macOS" in description, (
        "the macOS-only runner requirement must be documented in the "
        "action description so reuse on ubuntu-* fails on sight, not in CI"
    )


def test_diagnostics_neutralize_log_command_injection() -> None:
    composite = _load_composite()
    diagnostics = next(
        s for s in composite["runs"]["steps"] if s.get("name") == "Print new-code issues"
    )
    run = str(diagnostics["run"])
    assert 'gsub("[\\\\n\\\\r\\\\t]"; " ")' in run, (
        "issue messages echo content from the scanned (untrusted) tree: a "
        "newline plus ::error:: or ::add-mask:: in a message would be "
        "interpreted as a runner workflow command"
    )
    assert run.index("gsub") > run.index('"ISSUE '), (
        "the neutralizer must apply to the composed line, covering severity, "
        "rule, component and message fields alike"
    )
    assert "*[!0-9]*" in run, (
        "a non-numeric .total makes the -gt test error out instead of "
        "degrading gracefully like the rest of the step"
    )


def test_diagnostics_surface_failures_instead_of_swallowing() -> None:
    composite = _load_composite()
    diagnostics = next(
        s for s in composite["runs"]["steps"] if s.get("name") == "Print new-code issues"
    )
    run = str(diagnostics["run"])
    assert "set -o pipefail" in run
    assert "|| true" not in run, (
        "a swallowed curl/jq failure reads as 'no new-code issues': query "
        "failures must emit a visible warning"
    )
    assert "::warning::" in run
    assert "|| echo" in run, (
        "a 200 with an unexpected payload must degrade to a warning, not "
        "fail the job after a successful scan"
    )


def test_query_steps_urlencode_the_project_key() -> None:
    composite = _load_composite()
    for name in ("Print new-code issues",):
        step = next(s for s in composite["runs"]["steps"] if s.get("name") == name)
        run = str(step["run"])
        assert "curl -fsS -G" in run and "--data-urlencode" in run, (
            "the project key reaches the API as an encoded parameter, not "
            "raw URL interpolation that a ':' or '&' can corrupt"
        )


def test_callers_pass_only_declared_composite_inputs() -> None:
    declared = set(_load_composite()["inputs"])
    jobs = _load_workflow()["jobs"]
    for job in (jobs["sonarqube"], _dependabot_job(jobs)):
        for key in _composite_call(job).get("with", {}):
            assert key in declared, (
                f"composite call passes unrecognized input '{key}': GitHub "
                "silently ignores unknown action inputs, so a typo here "
                "disables the corresponding behavior without failing CI"
            )


def test_composite_rejects_whitespace_or_quotes_in_project_key() -> None:
    composite = _load_composite()
    step = next(s for s in composite["runs"]["steps"] if s.get("name") == "Validate inputs")
    run = str(step["run"])
    assert "[[:space:]" in run and "::error::" in run, (
        "SONAR_SCANNER_OPTS splits on whitespace: a project key containing "
        "whitespace or quotes must be rejected up front, not silently "
        "truncated at the scan invocation"
    )


def test_composite_rejects_empty_project_key() -> None:
    step = next(
        s for s in _load_composite()["runs"]["steps"] if s.get("name") == "Validate inputs"
    )
    run = str(step["run"])
    assert '-z "$PROJECT_KEY"' in run, (
        "required is only enforced when an input is omitted from with:: an "
        "empty expression passes and yields -Dsonar.projectKey= plus an "
        "unfiltered componentKeys= query instead of failing loudly"
    )
    assert run.index('-z "$PROJECT_KEY"') < run.index("[[:space:"), (
        "the emptiness check must come first: an empty key contains no "
        "whitespace and passes the character screen"
    )


def test_composite_rejects_multitoken_scanner_opts() -> None:
    step = next(
        s for s in _load_composite()["runs"]["steps"] if s.get("name") == "Validate inputs"
    )
    run = str(step["run"])
    assert "[[:space:][:cntrl:]]" in run, (
        "the scanner re-tokenizes SONAR_SCANNER_OPTS on whitespace: a second "
        "token could inject properties the blocklist does not know, so the "
        "guard must fail closed on any whitespace or control character"
    )


def test_composite_allowlists_scanner_opts() -> None:
    step = next(
        s for s in _load_composite()["runs"]["steps"] if s.get("name") == "Validate inputs"
    )
    assert str(step.get("env", {}).get("SCANNER_OPTS", "")) == "${{ inputs.scanner-opts }}"
    run = str(step["run"])
    assert "sonar\\.coverage\\." in run and "-Xmx" in run and "-Xss" in run, (
        "a blocklist is inherently incomplete (projectBaseDir, sources, "
        "exclusions, branch.name alter the scan materially): scanner-opts "
        "must be allowlisted to coverage properties and JVM sizes"
    )
    assert "sonar\\.(host\\.url|login|token|password)" not in run, (
        "the round-7 blocklist is superseded by the allowlist"
    )


def test_composite_validates_sonar_host_url_scheme() -> None:
    step = next(
        s for s in _load_composite()["runs"]["steps"] if s.get("name") == "Validate inputs"
    )
    assert str(step.get("env", {}).get("SONAR_HOST_URL", "")) == "${{ inputs.sonar-host-url }}"
    run = str(step["run"])
    assert "*@*" in run and "::error::" in run, (
        "the token rides the Bearer header to whatever host this input "
        "names: userinfo can redirect the effective host"
    )
    assert "http://127\\.0\\.0\\.1(:[0-9]+)?(/.*)?" in run, (
        "a glob arm like http://127.0.0.1* accepts 127.0.0.1.attacker.com: "
        "the loopback host must be anchored"
    )


def test_diagnostics_require_successful_validation() -> None:
    composite = _load_composite()
    validate = next(
        s for s in composite["runs"]["steps"] if s.get("name") == "Validate inputs"
    )
    assert validate.get("id") == "validate"
    diagnostics = next(
        s for s in composite["runs"]["steps"] if s.get("name") == "Print new-code issues"
    )
    assert "steps.validate.outcome == 'success'" in str(diagnostics.get("if", "")), (
        "!cancelled() runs the diagnostics even after a failed validation, "
        "sending the SONAR_TOKEN Bearer header to the host validation "
        "rejected"
    )


def test_composite_scan_wires_token_from_caller() -> None:
    scan = _step_with_name(_load_composite()["runs"], "SonarQube Scan")
    assert "${{ inputs.token }}" in str(scan.get("env", {}).get("SONAR_TOKEN", ""))
    assert not scan.get("id"), (
        "steps.scan was only read by the removed Enforce gate step: a dead id "
        "invites future steps to depend on a step outcome nothing guarantees"
    )


def test_dependabot_job_resets_scanner_config_to_trusted_main() -> None:
    composite = _load_composite()
    reset = next(
        (
            s
            for s in composite["runs"]["steps"]
            if s.get("name") == "Reset scanner config to trusted main"
        ),
        None,
    )
    assert reset is not None, (
        "the PR tree's sonar-project.properties must not steer the scan: "
        "reset it from trusted main before scanning"
    )
    assert "git show origin/main:sonar-project.properties" in str(reset["run"])
    assert "sonar-project.properties.tmp" in str(reset["run"]), (
        "the reset must be atomic: a truncated empty config must never "
        "replace the working file while the job continues"
    )
    assert "grep -q '[^[:space:]]' sonar-project.properties.tmp" in str(reset["run"]), (
        "grep -q . matches whitespace-only lines: a config of blank lines "
        "would pass and count as the trusted config"
    )
    assert "if ! git fetch --no-tags origin main" in str(reset["run"]), (
        "the fetch fallback runs without persisted credentials: a bare "
        "failure would be misreported by the following git show as a "
        "missing config on origin/main"
    )
    assert '"https://github.com/$GITHUB_REPOSITORY.git"' in str(reset["run"]), (
        "a substring remote match lets owner/repo-evil pass for owner/repo: "
        "the remote must equal the base repository's exact URL"
    )
    assert "::error::" in str(reset["run"]), (
        "this step is the security control keeping PR-sourced scan config "
        "away from the scan: a bare grep exit code is not a self-explanatory "
        "CI failure"
    )
    assert "rm -f sonar-project.properties.tmp" in str(reset["run"]), (
        "a failed reset must not leave the empty temp file behind in the "
        "workspace while the original file is untouched"
    )
    assert "git rev-parse --verify refs/remotes/origin/main" in str(reset["run"]), (
        "the reset fetches only when origin/main is not already local: the "
        "dependabot caller's restore step fetched it moments earlier"
    )
    assert "inputs.reset-config == 'true'" in str(reset.get("if", "")), (
        "composite inputs are strings and the string 'false' is truthy: the "
        "reset must compare against 'true' explicitly or it runs in every "
        "caller, including the pull_request job"
    )
    job = _dependabot_job(_load_workflow()["jobs"])
    assert str(_composite_call(job).get("with", {}).get("reset-config", "")).lower() == "true", (
        "the dependabot job must request the trusted-main config reset"
    )


def test_composite_declares_project_key_explicitly() -> None:
    composite = _load_composite()
    assert composite["inputs"]["project-key"]["required"] is True, (
        "the composite must not silently rely on caller job env: declare "
        "the project key as an input so a missing value fails loudly"
    )
    for name in ("Print new-code issues",):
        step = next(s for s in composite["runs"]["steps"] if s.get("name") == name)
        assert "${{ inputs.project-key }}" in str(
            step.get("env", {}).get("SONAR_PROJECT_KEY", "")
        )


def test_composite_single_sources_the_sonar_host_url() -> None:
    composite = _load_composite()
    assert composite["inputs"]["sonar-host-url"]["default"] == "http://127.0.0.1:9001"
    for name in (
        "Wait for SonarQube",
        "SonarQube Scan",
        "Print new-code issues",
    ):
        step = next(s for s in composite["runs"]["steps"] if s.get("name") == name)
        assert str(step.get("env", {}).get("SONAR_HOST_URL", "")) == (
            "${{ inputs.sonar-host-url }}"
        ), (
            "each step must take the SonarQube URL from the sonar-host-url "
            "input so the wait, scan and gate cannot drift to different endpoints"
        )
        assert "127.0.0.1" not in str(step.get("run", ""))


def test_jobs_pass_project_key_to_composite() -> None:
    jobs = _load_workflow()["jobs"]
    for job in (jobs["sonarqube"], _dependabot_job(jobs)):
        with_ = _composite_call(job).get("with", {})
        assert "env.SONAR_PROJECT_KEY" in str(with_.get("project-key", ""))


def test_dependabot_job_can_read_pull_requests() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    permissions = job["permissions"]
    assert permissions["contents"] == "read"
    assert permissions["pull-requests"] == "read"


def test_dependabot_job_resolves_pr_with_jq_and_fails_loudly() -> None:
    job = _dependabot_job(_load_workflow()["jobs"])
    resolve = _step_with_id(job, "pr")
    run = str(resolve["run"])
    assert "GITHUB_EVENT_PATH" in run
    assert "jq -r '.workflow_run.pull_requests[0].number // empty'" in run
    assert "EVENT_PR" not in run, (
        "the event PR number must not be evaluated as a GitHub expression: an empty "
        "pull_requests array must degrade to the API fallback, not fail the step"
    )
    assert "*[!0-9]*" in run, "resolved PR number must be validated as numeric"
    assert "pulls/$PR" in run
    assert ".user.login" in run, "the scan must verify the PR author is dependabot"
    assert '"$AUTHOR" != "dependabot[bot]"' in run, (
        "the author comparison must use the quoted canonical login: unquoted "
        "dependabot[bot] is a glob character class and never matches"
    )
    assert ".head.sha" in run, "the scan must verify the PR head SHA matches the scanned commit"
    assert "$HEAD_SHA" in run, "the head SHA comparison must use the validated workflow_run SHA"
    assert "curl -fsS" in run
    assert "|| true" not in run, "API errors must fail the job, not silently pass it"
    assert "grep" not in run, "parse API JSON with jq, not grep"


def test_pull_request_job_defers_dependabot_with_notice() -> None:
    job = _load_workflow()["jobs"]["sonarqube"]
    preflight = _step_with_id(job, "preflight")
    assert "github.event_name == 'pull_request'" in str(preflight.get("if", ""))
    run = str(preflight["run"])
    assert '"$PR_AUTHOR" = "dependabot[bot]"' in run
    text = WORKFLOW.read_text()
    assert "app/dependabot" not in text, (
        "use the canonical REST login only: app/dependabot is a GraphQL "
        "display form and must not appear in workflow logic"
    )
    assert "::notice::" in run
    assert 'echo "deferred=true" >> "$GITHUB_OUTPUT"' in run
    assert "*dependabot*" not in run, "substring author matching is too broad"


def test_pull_request_job_skips_scan_steps_when_deferred() -> None:
    job = _load_workflow()["jobs"]["sonarqube"]
    checkout = next(s for s in job["steps"] if "actions/checkout" in str(s.get("uses", "")))
    preflight = _step_with_id(job, "preflight")
    call = _composite_call(job)
    assert (
        job["steps"].index(preflight) < job["steps"].index(checkout) < job["steps"].index(call)
    )
    assert "steps.preflight.outputs.deferred != 'true'" in str(checkout.get("if", ""))
    assert "steps.preflight.outputs.deferred != 'true'" in str(call.get("if", "")), (
        "the composite call carrying every scan step must be gated by the "
        "preflight guard: none of the scan steps may run for deferred PRs"
    )


def test_workflow_uses_loopback_ip_not_localhost() -> None:
    for path in (WORKFLOW, COMPOSITE):
        assert "localhost:9001" not in path.read_text()


def test_diagnostic_steps_do_not_run_on_cancelled_jobs() -> None:
    for path in (WORKFLOW, COMPOSITE):
        assert "always()" not in path.read_text(), (
            "!cancelled() skips diagnostics on cancelled jobs"
        )


def test_checkout_actions_are_sha_pinned() -> None:
    steps = [s for job in _load_workflow()["jobs"].values() for s in job["steps"]]
    steps += _load_composite()["runs"]["steps"]
    for step in steps:
        uses = str(step.get("uses", ""))
        if uses.startswith("actions/checkout@"):
            assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", uses), (
                f"checkout must be SHA-pinned in this secret-bearing workflow: {uses}"
            )
