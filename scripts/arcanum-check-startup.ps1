Write-Output "=== Run key ==="
Get-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -ErrorAction SilentlyContinue |
    Format-List *

Write-Output "=== HKLM Run key ==="
Get-ItemProperty -Path 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run' -ErrorAction SilentlyContinue |
    Format-List *

Write-Output "=== Startup folder ==="
Get-ChildItem "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup" -ErrorAction SilentlyContinue

Write-Output "=== Scheduled tasks matching hermes ==="
Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object { $_.TaskName -match 'hermes' } |
    Select-Object TaskName, State

Write-Output "=== PowerShell execution policy ==="
Get-ExecutionPolicy -List

Write-Output "=== gcloud path ==="
Get-Command gcloud -ErrorAction SilentlyContinue | Select-Object Source

Write-Output "=== TEMP drive info ==="
Get-PSDrive C | Select-Object Used, Free
