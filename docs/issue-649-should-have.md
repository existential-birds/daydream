# Issue #649 — Should-have audit note

Should-have audit: grepped `daydream/improve/` for other `working_directory`-keyed dedup/lookup constructions; the only raw-spelling dedup site was `_host_enumerated_commands` (fixed in this PR). All other `working_directory` uses are schema/pass-through/render/host-emission and need no normalization.
