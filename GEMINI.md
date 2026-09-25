# GEMINI.md — Rules for Gemini CLI in This Repo

You are one contributor among several (a human maintainer and another AI agent) working on this repository. Your pull requests are held to a **higher review bar** than usual — expect thorough scrutiny before anything you open gets merged. Follow these rules exactly.

## Hard boundaries

- Never push to `main` directly. Always work on a new branch and open a pull request.
- Never merge your own pull request, approve it, or dismiss review comments as resolved yourself.
- Never modify GitHub repository settings, branch protection rules, secrets, or this file (`GEMINI.md`) itself.
- Never install or upgrade dependencies beyond what a task explicitly requires; don't re-scaffold the project or touch Docker/CI configuration unless the task is specifically about that.
- Stay confined to the scope of the issue you were asked to work on. If the task seems to require touching files clearly outside that scope, stop and explain why in a comment instead of proceeding.

## Branching & commits

- Branch naming: `gemini/<short-slug-of-the-issue>`.
- Commits are atomic: one logical change per commit, imperative mood subject line.
- No commented-out code, no debug prints, no unexplained TODOs.
- Every commit must leave the repo in a working state (imports/compiles, existing tests still pass).

## Pull requests

- PR description must state: what changed, why, what was explicitly NOT done, and exact commands to verify it.
- Label every PR you open with `source:gemini`.
- Link the GitHub issue you were working from.
- Keep PRs scoped to one concern — don't mix an unrelated fix into a feature PR.

## Code style

- Python backend: type hints on all function signatures. Pydantic models for all structured data (API schemas, any LLM-facing structured output). No untyped code.
- No dead code, no speculative abstractions — build only what the task requires.
- No comments explaining *what* code does; only comment non-obvious *why*.
- Fail loudly and explicitly. No silent `except: pass` or swallowed errors, especially anywhere touching data ingestion, citations, or answer generation — a silent failure there could produce a wrong answer that looks confident.
- TypeScript/React (if applicable): strict TypeScript, function components + hooks only, no class components.

## Testing

- The Python backend is managed with `uv` (`backend/pyproject.toml` + `uv.lock`). Use `uv run pytest` to run tests, `uv add`/`uv add --dev` to add a dependency your task actually needs.
- Every new module (ingestion source, retriever, provider, endpoint) needs at least one test using mocked HTTP/fixtures — never a live network call or a live LLM call in tests.
- Run the full test suite yourself and confirm it passes before opening the PR. State the exact command and result in the PR description.

## Trust & accuracy rules specific to this project

- This project answers German tax/legal/business-registration questions. Any code path that produces or transforms an answer to the user must preserve source citations (URL, statute/section, retrieval date) — never drop or fabricate provenance data.
- Do not weaken, remove, or work around any existing validation that flags an uncited or low-confidence answer as unverified.

## If you're unsure

Stop and leave a comment on the issue explaining what's unclear rather than guessing and proceeding.
