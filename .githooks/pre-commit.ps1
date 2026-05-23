#Requires -Version 5.1
<#
.SYNOPSIS
  Block commits that stage secrets or credential-like content.

  Bypass (only if you are certain): git commit --no-verify
  WARNING: --no-verify skips secret scanning and can leak keys to git history.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-HookError {
    param([string]$Message)
    Write-Host ""
    Write-Host "PRE-COMMIT SECRET SCAN FAILED" -ForegroundColor Red
    Write-Host $Message -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Fix the staged content, unstage secrets, or use placeholders in .env.example only."
    Write-Host "To bypass (NOT recommended): git commit --no-verify"
    Write-Host "Bypassing can permanently leak credentials into git history."
    Write-Host ""
    exit 1
}

$staged = git diff --cached --name-only --diff-filter=ACMRT 2>$null
if (-not $staged) { exit 0 }

$blockedNames = @('\.env$', '\.env\..+$', '\.key$', '\.pem$', '\.p12$', '\.pfx$', '^secrets\.', 'fred_key\.txt$')
foreach ($path in $staged) {
    $leaf = Split-Path $path -Leaf
    foreach ($pat in $blockedNames) {
        if ($leaf -match $pat -and $leaf -ne '.env.example') {
            Write-HookError "Blocked staged secret file: $path"
        }
    }
}

$placeholderFred = '^(?i)(your_fred_api_key_here|your_api_key_here|<fred_api_key_here>|changeme|replace_me|xxx\.?\.?\.?|placeholder)$'
$fredReal = '(?i)^FRED_API_KEY\s*=\s*(?!your_fred_api_key_here|your_api_key_here|<fred_api_key_here>|changeme|replace_me|xxx|placeholder)([0-9a-fA-F]{32})\s*$'
$secretHexLine = '(?i)(API_KEY|SECRET|TOKEN|PASSWORD).{0,20}([0-9a-fA-F]{32})'
$awsKey = 'AKIA[0-9A-Z]{16}'

$added = git diff --cached -U0 --no-color
if (-not $added) { exit 0 }

foreach ($line in ($added -split "`n")) {
    if ($line -notmatch '^\+' -or $line -match '^\+\+\+') { continue }
    $content = $line.Substring(1)

    if ($content -match $fredReal) {
        Write-HookError "Staged diff contains a real-looking FRED_API_KEY value.`nLine: $content"
    }

    if ($content -match $secretHexLine) {
        $val = $Matches[2]
        if ($val -notmatch $placeholderFred) {
            Write-HookError "Staged diff contains a 32-char hex value near a secret keyword.`nLine: $content"
        }
    }

    if ($content -match $awsKey) {
        Write-HookError "Staged diff contains an AWS access key pattern (AKIA...).`nLine: $content"
    }
}

exit 0
