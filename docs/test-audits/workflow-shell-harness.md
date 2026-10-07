# Workflow match harness correction

Pinned baseline: `6c48d368d529a89748c93e7bb61ae9ac6566936e`.
Collection after repair: **8,200 cases / 5,360 functions**. Net reduction from baseline: **804 cases (8.93%) / 71 functions**. Three new regression cases are an explicit addition, not a deletion quota. Prior measured resume head `3078954b` collected 8,197 cases.

## Retained failure and actual owner

[CI 37567336935](https://github.com/existential-birds/daydream/actions/runs/37567336935) failed at the unchanged node `tests/test_workflow_templates.py::test_match_step_recognizes_exactly_the_three_bot_commands[flowchart-mid-body-template-command]`: actual output command was empty, expected flowchart. Result: one failed / 8,191 passed / five skipped / 73 warnings in 361.10s, coverage 90.55%, Python 3.12.14. The original short input and assertion remain unchanged. No retry, skip or workflow edit hides it.

The helper executed each shipped match step with `bash --noprofile --norc -eo pipefail -c`. All three complete source owners have unspecified shell at workflow, job and step levels:

- `.github/workflows/daydream-command.yml`, dispatch match step;
- `daydream/templates/workflows/daydream-command.yml`, dispatch match step;
- `daydream/templates/workflows/single/daydream.yml`, gate match step.

GitHub's actual unspecified shell is **`bash -e {0}`**. Explicit `shell: bash` selects the different pipefail command. [Primary shell documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstepsshell). History `10e319a7` (#1113/#1120) introduced the real execution matrix without declaring pipefail in those shipped match steps. Full test, fixture/output helper, three source owners, installation caller `daydream/bot_setup.py`, shell settings and relevant history were read before editing. The affected seam has only the existing match matrix as caller; other workflow/package contracts remain.

## Regression control and limits

The unchanged actual helper returned wrong empty command in 10/10 Darwin runs with a valid flowchart mention followed by 30,000 globe Unicode characters. On Linux, the same unchanged shipped script/input failed 99/100 under the forced pipefail policy and 0/100 with the declared default. A large producer can encounter grep's successful early read termination; the forced pipeline policy treats that differently from the deployed policy.

The exact short CI event's cause remains an inference: no PIPESTATUS/signal/write trace was captured and 1,500 short Linux replays did not fail. The deterministic larger input proves a harness policy defect independently. Do not claim the control directly traces the original CI scheduling event.

Added exact nodes, all in the existing declaration:

- `test_match_step_recognizes_exactly_the_three_bot_commands[flowchart-large-unicode-body-live-command]`
- `test_match_step_recognizes_exactly_the_three_bot_commands[flowchart-large-unicode-body-template-command]`
- `test_match_step_recognizes_exactly_the_three_bot_commands[flowchart-large-unicode-body-single]`

All three fail against the original helper at the intended command assertion (three failed in 0.66s), then pass as part of the corrected full matrix. The input has 30,032 Unicode characters and remains environment data; it is not interpolated into shell source.

## Bounded repair and keepers

Change only test-local `_run_match_step`: derive workflow/job/step effective shell and assert it remains unspecified; write exact step bytes to a real isolated temporary script; execute `bash -e` against that file. Keep PATH/BODY/BOT_HANDLE/GITHUB_OUTPUT isolation, real subprocess, check=True, captured streams and actual output-file decoding. Future explicit shell selection fails visibly and requires updating the harness. No production parser, copied regex, fake output or retry is introduced.

All prior 25 input rows retain identical ASTs/order and both observable command/matched assertions across all three shipped source owners: 75 existing cases remain; the added input gives 78. Vocabulary, aliases, whitespace/newline/CRLF, whole-body malformed-sequence veto, precedence, literal regex-metacharacter handles, case/plural/word boundaries and negative commands retain their proof. Approval/head/time binding, injection, action pins, permissions, acknowledgement ordering and installation contracts remain unchanged.

Independent read-only plan/applied diff/AST review approved the repair. No function added or deleted; only helper and parameter table change. Shipped workflows, CI, production, locks, runner size, coverage and skip settings remain byte-identical.

Validation: workflow plus command-contract consumers **193 passed, 31.09s pytest / 31.45s wall**; bot setup **24 passed, 2.01s**; collection **8,200 / 5,360**. An earlier mistyped consumer path collected zero tests and supplies no validation proof. Local cleanup/startup cost means these focused elapsed times are not a before/after performance claim. Final ordinary make check/push and hosted result are recorded in the PR after actual completion.
