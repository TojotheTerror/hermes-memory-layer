Get-Process Hermes -ErrorAction SilentlyContinue | Select-Object Id, StartTime, Path
Write-Output '---SHORTCUT-TARGET---'
$sh = New-Object -ComObject WScript.Shell
$sc = $sh.CreateShortcut('C:\Users\rpgmo\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Hermes.lnk')
Write-Output "Target: $($sc.TargetPath)"
Write-Output "Args: $($sc.Arguments)"
Write-Output '---BACKUP-SHORTCUTS-PRESENT---'
Get-ChildItem 'C:\Users\rpgmo\AppData\Roaming\Microsoft\Windows\Start Menu\Programs' -Filter 'Hermes.lnk.bak*'
Write-Output '---SCHEDULED-TEST-TASK-CLEANUP-CHECK---'
Get-ScheduledTask -TaskName A4LaunchTest -ErrorAction SilentlyContinue | Select-Object TaskName, State
