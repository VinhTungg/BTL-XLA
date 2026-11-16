param(
    [ValidateSet('cifar100', 'caltech101')]
    [string]$Dataset = 'cifar100',
    [ValidateSet('microbatch', 'effective_batch', 'full100')]
    [string]$Preset,
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
    $demoArgs = @('-X', 'utf8', '-m', 'tacs.showcase', '--dataset', $Dataset, '--device', $Device)
    if ($Preset) { $demoArgs += @('--preset', $Preset) }
    & $pythonExe @demoArgs
    if ($LASTEXITCODE -ne 0) { throw "Demo dừng với mã $LASTEXITCODE; xem lỗi ở trên." }
}
finally { Pop-Location }
