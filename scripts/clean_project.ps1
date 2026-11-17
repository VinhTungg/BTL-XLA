[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Apply,
    [switch]$Json
)

# Default: read-only inventory. Deletion requires the explicit -Apply switch.
# Only the three Git temporary files approved on 2026-11-17 are eligible.
$ErrorActionPreference = 'Stop'
if ($Json) { [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false) }
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$projectPrefix = $projectRoot.TrimEnd([char[]]'\/') + [IO.Path]::DirectorySeparatorChar
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'pyproject.toml') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $projectRoot 'src\tacs') -PathType Container)) {
    throw 'The script must be inside the TACS project scripts directory.'
}

function Assert-SafeTarget {
    param([string]$Target)
    $absolute = [IO.Path]::GetFullPath($Target)
    if (-not $absolute.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        $absolute -eq $projectRoot) {
        throw "Target outside the project: $absolute"
    }
    $item = Get-Item -LiteralPath $absolute -Force
    $ancestor = $item.Directory
    if ($item.PSIsContainer) { $ancestor = $item.Parent }
    while ($null -ne $ancestor -and $ancestor.FullName -ne $projectRoot) {
        if (($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing a target below a symlink or junction: $absolute"
        }
        $ancestor = $ancestor.Parent
    }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing a symlink or junction: $absolute"
    }
    if ($item.PSIsContainer) {
        $links = @(Get-ChildItem -LiteralPath $absolute -Recurse -Force -Attributes ReparsePoint)
        if ($links.Count -gt 0) {
            throw "Refusing a directory containing symlinks or junctions: $absolute"
        }
    }
    return $item
}

$relativeTargets = @('.venv-clean', '.pytest_cache', 'tmp\pdfs', 'tmp\uv-cache', '.git-initialized-backup')
foreach ($codeDirectory in @('src', 'scripts', 'tests')) {
    $codePath = Join-Path $projectRoot $codeDirectory
    if (Test-Path -LiteralPath $codePath -PathType Container) {
        foreach ($cache in Get-ChildItem -LiteralPath $codePath -Directory -Recurse -Force -Filter '__pycache__') {
            $relativeTargets += $cache.FullName.Substring($projectPrefix.Length)
        }
    }
}
$relativeTargets += @(
    '.git\objects\8c\tmp_obj_jLGv0Z',
    '.git\objects\8c\tmp_obj_WHH6O0',
    '.git\objects\96\tmp_obj_RDE2Fw'
)

$plan = @()
foreach ($relative in $relativeTargets) {
    $candidate = Join-Path $projectRoot $relative
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    $item = Assert-SafeTarget $candidate
    $files = if ($item.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $item.FullName -Force -Recurse -File)
    } else { @($item) }
    if ($relative -eq '.git-initialized-backup' -and $files.Count -ne 0) {
        throw 'The obsolete Git backup is not empty; keep it and inspect manually.'
    }
    if ($relative -eq '.venv-clean' -and (
        -not (Test-Path -LiteralPath (Join-Path $candidate 'pyvenv.cfg') -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $projectRoot '.venv\Scripts\python.exe') -PathType Leaf))) {
        throw 'Old venv validation failed, or the primary .venv is missing.'
    }
    $plan += [PSCustomObject]@{
        RelativePath = $relative
        AbsolutePath = $item.FullName
        Files = $files.Count
        Bytes = [long]($files | Measure-Object Length -Sum).Sum
        Status = 'planned'
    }
}

$summary = [PSCustomObject]@{
    Mode = 'dry_run'
    ProjectRoot = $projectRoot
    Targets = @($plan)
    TotalBytes = [long]($plan | Measure-Object Bytes -Sum).Sum
    Protected = @('.venv', 'data', 'artifacts', 'outputs', 'cifar100', 'src', 'tests', 'docs', 'Git commits and objects')
}
if (-not $Apply -or $WhatIfPreference) {
    if ($Json) { $summary | ConvertTo-Json -Depth 6 } else {
        $plan | Select-Object RelativePath, Files, @{Name='MiB';Expression={[math]::Round($_.Bytes / 1MB, 2)}} | Format-Table -AutoSize
        Write-Output ('Potential cleanup: {0:N2} GiB. No files deleted. Run with -Apply to delete this exact plan.' -f ($summary.TotalBytes / 1GB))
    }
    return
}

if (Get-Process -Name git, uv -ErrorAction SilentlyContinue) {
    throw 'Git or uv is running. Stop it before applying cleanup.'
}
$oldVenv = Join-Path $projectRoot '.venv-clean'
if ($env:VIRTUAL_ENV -and [IO.Path]::GetFullPath($env:VIRTUAL_ENV).TrimEnd([char[]]'\/') -eq $oldVenv) {
    throw 'The shell is using .venv-clean. Deactivate it before cleanup.'
}
$oldVenvProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ExecutablePath -and $_.ExecutablePath.StartsWith($oldVenv + '\', [StringComparison]::OrdinalIgnoreCase)
})
if ($oldVenvProcesses.Count -gt 0) {
    throw 'An executable from .venv-clean is running; keep the environment.'
}

# Validate all targets again before making the first deletion.
foreach ($entry in $plan) { $null = Assert-SafeTarget $entry.AbsolutePath }
$reportDirectory = Join-Path $projectRoot 'artifacts\cleanup'
New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
$reportPath = Join-Path $reportDirectory ('cleanup-{0}-{1}.json' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), [guid]::NewGuid().ToString('N'))
$summary.Mode = 'applying'
$summary | Add-Member -NotePropertyName ReportPath -NotePropertyValue $reportPath
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding UTF8
try {
    foreach ($entry in $plan) {
        if ($PSCmdlet.ShouldProcess($entry.AbsolutePath, 'Delete the listed disposable project cache or approved Git temporary file')) {
            $null = Assert-SafeTarget $entry.AbsolutePath
            Remove-Item -LiteralPath $entry.AbsolutePath -Recurse -Force -ErrorAction Stop
            if (Test-Path -LiteralPath $entry.AbsolutePath) { throw "Residual target: $($entry.AbsolutePath)" }
            $entry.Status = 'removed'
        } else { $entry.Status = 'skipped' }
        $summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding UTF8
    }
    $summary.Mode = 'complete'
} catch {
    $summary.Mode = 'failed_partial'
    $summary | Add-Member -NotePropertyName Error -NotePropertyValue $_.Exception.Message
    throw
} finally {
    $summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding UTF8
}
if ($Json) { $summary | ConvertTo-Json -Depth 6 } else {
    $removed = [long]($plan | Where-Object Status -eq 'removed' | Measure-Object Bytes -Sum).Sum
    Write-Output ('Deleted {0:N2} GiB of listed files. Report: {1}' -f ($removed / 1GB), $reportPath)
    Write-Output 'Caches and the secondary venv can be rebuilt. Git commits, model artifacts and datasets were kept.'
}
