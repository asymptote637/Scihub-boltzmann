param(
    [string]$ShortcutName = "Scihub Boltzmann.lnk"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Target = Join-Path $Root "start_lbm_desktop.bat"

if (-not (Test-Path $Target)) {
    throw "Launcher not found: $Target"
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop $ShortcutName

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Target
$Shortcut.WorkingDirectory = $Root
$Shortcut.Description = "Open the Scihub Boltzmann 2D LBM desktop simulator"
$Shortcut.Save()

Write-Host "Created desktop shortcut: $ShortcutPath"
