param([switch]$Autostart)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = $null
foreach ($version in "3.12", "3.11", "3.10") {
    try { & py "-$version" -c "import tkinter" 2>$null; if ($LASTEXITCODE -eq 0) { $python = @("py", "-$version"); break } } catch {}
}
if (-not $python) {
    Write-Error "Python 3.10-3.12 is needed. Install it from python.org (with Tk), then run this again."
}

Write-Host "==> Python packages"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $python[0] $python[1] -m venv .venv }
& .venv\Scripts\python.exe -m pip install --quiet --upgrade pip
& .venv\Scripts\python.exe -m pip install --quiet -r requirements.txt

Write-Host "==> Hand tracking model"
New-Item -ItemType Directory -Force models, config, logs | Out-Null
if (-not (Test-Path "models\hand_landmarker.task")) {
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
        -OutFile "models\hand_landmarker.task.part"
    Move-Item -Force "models\hand_landmarker.task.part" "models\hand_landmarker.task"
}

function New-Shortcut($path, $arguments) {
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
    $link.Arguments = "`"$(Join-Path $PSScriptRoot 'app.py')`" $arguments"
    $link.WorkingDirectory = $PSScriptRoot
    $link.Description = "Type into any app by tapping the table"
    $link.Save()
}

New-Shortcut (Join-Path ([Environment]::GetFolderPath("Programs")) "0Keys.lnk") ""
Write-Host "==> Added 0Keys to the Start menu"
if ($Autostart) {
    New-Shortcut (Join-Path ([Environment]::GetFolderPath("Startup")) "0Keys.lnk") "--minimized"
    Write-Host "==> 0Keys will start when you log in"
}

Write-Host ""
Write-Host "Done. Open 0Keys from the Start menu, or run 0keys.bat."
Write-Host "First click 'Set up keyboard', then press Ctrl+Alt+K in any app to type."
