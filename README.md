# Kanzlei-KI

A source-cited, human-verified AI assistant for German Steuerberater/Finanz/Legal questions — starting with a narrow, personally-validatable MVP: **registering as a Freiberufler or starting a GbR in Germany**.

## Why

Generic AI answers on tax/legal questions can be confidently wrong, and wrong is expensive here. Kanzlei-KI is built around one hard rule: **every answer must cite an official source (a statute paragraph, an official form, a government guidance page), or it's flagged as unverified.** Answers are logged and reviewed by a human before they're trusted, and that review process builds a growing "golden" set of verified questions and answers used to measure and gate accuracy over time.

## Scope (MVP)

- **Knowledge backend**: official German sources (statutes from gesetze-im-internet.de, BMWK's existenzgruender.de founder guidance, public ELSTER/Finanzamt guidance) ingested, chunked, and made retrievable with full source metadata.
- **Chat**: ask a question, get an answer with inline citations — or an explicit "unverified, confirm manually" flag if no reliable citation was found.
- **Review queue**: every answer is reviewable and correctable, feeding a golden evaluation set used to measure accuracy before anything is trusted as "final."
- Structured document analysis (e.g. uploading a contract or a Gewerbeanmeldung form for review) is a later milestone, once the chat accuracy bar is met.

## Stack

Self-hostable, Docker-first, permissively-licensed open-source components: FastAPI, Qdrant, Ollama (with an OpenRouter cloud fallback for testing bigger models), Next.js.

## Status

See the [Milestones](../../milestones) for the current phase breakdown.

## License

Source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). Free to use, study, and self-host for noncommercial purposes. Commercial use requires a separate license from the author.
