Write-Output "=== WSL systemd support (Ubuntu-24.04) ==="
wsl -d Ubuntu-24.04 -e bash -c "cat /etc/os-release | head -3; ps -p 1 -o comm="

Write-Output "=== Hermes.exe process tree ==="
Get-CimInstance Win32_Process -Filter "Name='Hermes.exe'" -ErrorAction SilentlyContinue |
    Select-Object ProcessId, ParentProcessId, CommandLine

Write-Output "=== Parent of Hermes.exe (who launched it) ==="
$h = Get-CimInstance Win32_Process -Filter "Name='Hermes.exe'" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($h) {
    Get-CimInstance Win32_Process -Filter "ProcessId=$($h.ParentProcessId)" -ErrorAction SilentlyContinue |
        Select-Object ProcessId, Name, CommandLine
}

Write-Output "=== python.exe gateway process env check ==="
$py = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'hermes_cli' }
$py | Select-Object ProcessId, CommandLine
