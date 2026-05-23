# Security — Secret Handling Protocol

This document describes how this repository stores, validates, and protects API keys and other credentials.

## 1. Where keys are stored

| Secret | Location | Committed? |
|--------|----------|------------|
| FRED API key | `.env` (`FRED_API_KEY=...`) or file via `FRED_API_KEY_FILE` | **Never** |
| IBKR account | `.env` (`IBKR_ACCOUNT=...`) or process env | **Never** |
| IB Gateway host/port/client | `config/paper_trading.yaml` + optional env overrides | Config yes; credentials no |

- Copy `.env.example` → `.env` and fill in real values locally.
- `.env`, `.env.*` (except `.env.example`), `*.key`, `fred_key.txt`, and `secrets/` are gitignored.
- Do **not** store keys in source code, notebooks, logs, or chat/screenshots.

## 2. Runtime validation

`src/utils/fred_key.py` exposes `validate_fred_key_format()` and calls it from `get_fred_api_key()` before returning a resolved key.

Validation rules:

- Raises `ValueError` if the key is missing, empty, or a known placeholder (`your_fred_api_key_here`, `changeme`, `REPLACE_ME`, etc.).
- Raises `ValueError` if length is not exactly 32 characters.
- Logs a warning if the key is not lowercase hexadecimal (expected FRED format).

## 3. Commit protection

Two layers:

1. **Git hook** (`.githooks/pre-commit.ps1`, enabled via `git config core.hooksPath .githooks`):
   - Blocks staged `.env`, `*.key`, `*.pem`, `fred_key.txt`, and similar paths.
   - Scans staged diffs for real-looking `FRED_API_KEY=` values, 32-char hex near secret keywords, and AWS `AKIA...` patterns.
   - Prints how to bypass with `--no-verify` and warns that bypassing risks permanent history leaks.

2. **pre-commit framework** (`.pre-commit-config.yaml`):
   - **gitleaks** — general secret scanner.
   - **local `block-secrets-pre-commit`** — runs the same PowerShell hook.

Install/update hooks:

```powershell
git config core.hooksPath .githooks
uv run pre-commit install
```

## 4. CI / GitHub Actions

There is currently **no** `.github/workflows/` in this repository.

When CI is added:

- Reference secrets only as `${{ secrets.FRED_API_KEY }}` (or GitHub Encrypted Secrets equivalent).
- Do **not** `echo`, `printenv`, or `env` dump variables in workflow steps.
- Do **not** upload raw logs as artifacts without filtering key material.

## 5. Rotation policy

Rotate credentials **immediately** when:

- A key appears in git history, logs, chat, email, or a screenshot.
- A developer bypasses pre-commit with `--no-verify` to commit env-like content.
- A former collaborator had access and may have copied keys.
- FRED or IBKR reports unusual API usage.

FRED key rotation: https://fred.stlouisfed.org/docs/api/api_key.html

After rotation, update local `.env` only — never commit the new value.

## 6. Known git history exposure

Git history contains a **hardcoded FRED API key** in early commits (removed from current tree):

| Commit | Date | File |
|--------|------|------|
| `c337aa7` | 2026-02-03 | `economic_regime_with_fred.py` |
| `adb6899` | 2026-02-20 | `src/economic_regime.py` |
| `070a3f0` | 2026-02-22 | `src/economic_regime.py` |

**Action required:** Rotate the FRED key if that key was ever active or pushed to a remote.

To scrub history (rewrites all commits — coordinate with collaborators):

```bash
# Example using git-filter-repo (install separately)
git filter-repo --invert-paths --path economic_regime_with_fred.py  # or use --replace-text
# Or BFG Repo-Cleaner: https://rtyley.github.io/bfg-repo-cleaner/
```

Force-push after scrubbing: `git push --force-with-lease` (warn all collaborators).

## 7. If a leak is discovered

1. **Rotate** the exposed credential immediately (treat as compromised).
2. **Remove** the secret from working tree and unstage any commits that contain it (do not push further).
3. **Scrub history** if the value was committed or pushed (`git-filter-repo` / BFG).
4. **Audit** logs under `logs/` and archived reports under `archive/` for fingerprints.
5. **Document** the incident (date, scope, rotation confirmation) outside the repo if required by policy.

## 8. Log retention

- Keep at most **7 days** of active logs under `logs/`.
- Archive older logs to `archive/` (never commit).
- Never log key values — not even masked — in application code. Diagnostic scripts may show masked fingerprints only; do not paste that output externally.

## 9. Developer checklist

- [ ] `.env` exists locally, `.env.example` uses placeholders only
- [ ] `git config core.hooksPath .githooks` is set for this clone
- [ ] `python scripts/_check_fred_key.py` reports `API test: OK`
- [ ] No `git commit --no-verify` for env/credential files
- [ ] Screenshots exclude terminal key-resolution sections
