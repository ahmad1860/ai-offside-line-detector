# Data directory

This project **does not scrape or auto-download football footage**. You supply your own clips.

## Layout

- `raw/` — your input clips (e.g. `.mp4`). **Gitignored** — never committed.
- `samples/` — tiny, non-copyrighted assets used by the automated tests (e.g. short
  programmatically generated clips). Safe to commit.

## Footage rights (please read)

Only use footage you have the right to use:

- Clips you recorded yourself.
- Material you are licensed to use, or that is clearly released under a permissive/open licence.
- Public-domain or Creative-Commons footage, respecting the specific licence terms.

Do **not** add copyrighted broadcast footage to this repository. Keep large media out of Git;
`raw/` is gitignored for this reason.

## Clip guidance (CPU-first)

For reasonable performance on a CPU-only laptop:

- Keep clips short (a few seconds up to ~15 s).
- The pipeline samples frames and resizes them (see `config/default.yaml`) to stay within
  limited RAM and CPU budgets.
