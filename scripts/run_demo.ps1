param(
    [ValidateSet('microbatch', 'effective_batch')]
    [string]$Preset = 'microbatch',
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Không tìm thấy Python của project: $pythonExe"
}
Push-Location -LiteralPath $projectRoot
try {
    & $pythonExe -X utf8 -m tacs.showcase --preset $Preset --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Demo dừng với mã $LASTEXITCODE; xem lỗi ở trên." }
}
finally { Pop-Location }
