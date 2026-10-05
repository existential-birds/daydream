# Capture run evidence at finalization and cooperative interruption

For the new JSONL evidence workflow in [issue #1467](https://github.com/existential-birds/daydream/issues/1467), capture uses the existing frozen run snapshot at finalization and cooperative interruption. We accept that SIGKILL or power loss can lose evidence that has not reached capture, keeping incremental durable event recording outside this ticket's scope. Local persistence must still use atomic, recoverable writes and diagnose failures honestly; an interrupted write must never appear to be a complete published record.
