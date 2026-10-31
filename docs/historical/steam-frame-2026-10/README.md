# Historical Steam Frame stereo / tracing experiment

Base: fc48b6155ebb4fd658ddb8f8b2006086659747e4. Archived 2026-10-31.

This branch preserves the old separate-eye/controller diagnostic stack, the
tracing feedback argument-order fix and opt-in sender pacing. It is not a
proposal for the current development branch. Full builds and device tests
were not rerun during archival; git diff --check passed.

The Monado dependency is the monado-rev pin plus the 12 patches saved here.
The exact historical tree and pin are in monado-provenance.json. Raw runtime
logs and measurement reports are kept separately as investigation records.
