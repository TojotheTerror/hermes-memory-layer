# Standalone script to repoint Arcanum's Hermes Start Menu shortcut at the
# new launch wrapper. Run once, locally on Arcanum, via SSH.
#
# Spec ref: docs/sietch-arcanum-implementation-spec.md §2.3 Step A2
# Written as a standalone .ps1 (not inline PowerShell-over-SSH) to avoid
# the recurring nested-quoting bug documented in this project's lessons
# learned.

$ErrorActionPreference = "Stop"

$shortcutPath = "C:\Users\rpgmo\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Hermes.lnk"
$wrapperPath  = "C:\Users\rpgmo\hermes-memory-layer\scripts\arcanum-launch-hermes.ps1"

if (-not (Test-Path $shortcutPath)) {
    Write-Error "Shortcut not found at expected path: $shortcutPath"
    exit 1
}
if (-not (Test-Path $wrapperPath)) {
    Write-Error "Wrapper script not found at expected path: $wrapperPath (ship it first)"
    exit 1
}

# Back up the original shortcut before modifying it.
$backupPath = "$shortcutPath.bak-$(Get-Date -Format yyyyMMddHHmmss)"
Copy-Item -Path $shortcutPath -Destination $backupPath -Force
Write-Output "Backed up original shortcut to: $backupPath"

$sh = New-Object -ComObject WScript.Shell
$shortcut = $sh.CreateShortcut($shortcutPath)

Write-Output "Original Target:    $($shortcut.TargetPath)"
Write-Output "Original Arguments: $($shortcut.Arguments)"

$originalTarget = $shortcut.TargetPath
$originalArgs   = $shortcut.Arguments
$originalIcon   = $shortcut.IconLocation

$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments  = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$wrapperPath`""
# Preserve the original icon (falls back to Hermes.exe's icon) so the
# Start Menu entry still looks like Hermes, not a generic PowerShell icon.
if ($originalIcon -and $originalIcon -ne ",0") {
    $shortcut.IconLocation = $originalIcon
} elseif ($originalTarget) {
    $shortcut.IconLocation = "$originalTarget,0"
}
$shortcut.Save()

Write-Output "New Target:    $($shortcut.TargetPath)"
Write-Output "New Arguments: $($shortcut.Arguments)"
Write-Output "New Icon:      $($shortcut.IconLocation)"
Write-Output "Shortcut updated successfully."
