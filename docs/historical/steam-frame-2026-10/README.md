# Historical Steam Frame experiment

Archived on 2026-10-31 from the temporary Codex workspace.
Base commit: `fc48b6155ebb4fd658ddb8f8b2006086659747e4`.

This branch preserves old diagnostic work, not a proposal for the current
WiVRn development branch. Full builds and device tests were not rerun during
archival. `git diff --check` passed.

Sender pacing is opt-in with WIVRN_VIDEO_PACING_BPS. The historical corrected
fail-open behavior is preserved. A successful past build does not establish
current runtime compatibility.
