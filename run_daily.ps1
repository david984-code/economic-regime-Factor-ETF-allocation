# Daily Portfolio Update - PowerShell Script
# Enhanced version with error handling and notifications

param(
    [switch]$SendEmail = $false,
    [string]$EmailTo = ""
)

# Change to script directory
Set-Location $PSScriptRoot

# Run the Python script
Write-Host "Starting daily portfolio update at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "=" * 80

$process = Start-Process -FilePath "python" -ArgumentList "run_daily_update.py" -NoNewWindow -Wait -PassThru

$exitCode = $process.ExitCode

if ($exitCode -eq 0) {
    Write-Host "`n✓ Portfolio update completed successfully" -ForegroundColor Green
    
    # Optional: Send email notification
    if ($SendEmail -and $EmailTo) {
        # Configure your SMTP settings here
        # Send-MailMessage -To $EmailTo -From "your@email.com" -Subject "Portfolio Update: Success" -Body "Daily update completed at $(Get-Date)" -SmtpServer "smtp.gmail.com"
    }
} else {
    Write-Host "`n✗ Portfolio update failed with exit code $exitCode" -ForegroundColor Red
    
    # Optional: Send alert email
    if ($SendEmail -and $EmailTo) {
        # Send-MailMessage -To $EmailTo -From "your@email.com" -Subject "Portfolio Update: FAILED" -Body "Check logs for details. Exit code: $exitCode" -SmtpServer "smtp.gmail.com" -Priority High
    }
}

Write-Host "`nLog file: logs\daily_update_$(Get-Date -Format 'yyyyMMdd').log"

exit $exitCode
