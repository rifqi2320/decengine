# System One wire compatibility

The repository freezes the known v1 contract at `systemone-v1`:

- request fields: `model`, JSON `state`, and named `questions`;
- question discriminators: `choice`, `noul`, and `score`;
- typed result objects plus request ID, model, creation time, and usage;
- a stable `{ "error": { "code", "message" } }` envelope;
- model discovery at `GET /v1/models`.

Fixtures live in `tests/protocol/fixtures`. The dependency-free black-box client in
`tests/uat/systemone_black_box.py` represents the consumer boundary.

The PRD requires an existing System One SDK to be the authoritative release test. No SDK
or canonical public schema was supplied with the source requirements, so this repository
does not claim that the included inferred fixture proves that external compatibility.
Before release, capture a request/response transcript from the approved SDK, add it as a
sanitized fixture, and run that SDK in CI against `decengine serve`. Preserve the internal
C ABI JSON as a separate protocol even if fields overlap.
