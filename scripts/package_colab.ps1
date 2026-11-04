param(
    [string]$OutputPath = "artifacts\TACS_Colab_Bundle.zip",
    [switch]$WithoutCheckpoint
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$outputFile = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    [System.IO.Path]::GetFullPath($OutputPath)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
}
$outputParent = Split-Path -Parent $outputFile
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null

$tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$staging = Join-Path $tempRoot ("tacs-colab-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $staging | Out-Null

try {
    foreach ($directory in @("src", "scripts", "tests", "notebooks")) {
        Copy-Item -LiteralPath (Join-Path $repoRoot $directory) -Destination $staging -Recurse
    }
    foreach ($file in @("pyproject.toml", "README.md", "uv.lock")) {
        Copy-Item -LiteralPath (Join-Path $repoRoot $file) -Destination $staging
    }
    $docsDestination = Join-Path $staging "docs"
    New-Item -ItemType Directory -Path $docsDestination -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot "docs\CONTEXT_ROLE_FIX.md") -Destination $docsDestination

    $splitSource = Join-Path $repoRoot "data\splits\cifar100_seed42.json"
    $splitDestination = Join-Path $staging "data\splits"
    New-Item -ItemType Directory -Path $splitDestination -Force | Out-Null
    Copy-Item -LiteralPath $splitSource -Destination $splitDestination

    if (-not $WithoutCheckpoint) {
        $relativeRun = "outputs\cifar100\dinov3_vits16\seed_42\no_context"
        $runSource = Join-Path $repoRoot $relativeRun
        $runDestination = Join-Path $staging $relativeRun
        if (-not (Test-Path -LiteralPath $runSource -PathType Container)) {
            throw "Checkpoint directory not found: $runSource"
        }
        New-Item -ItemType Directory -Path $runDestination -Force | Out-Null
        foreach ($name in @(
            "checkpoint_best.pt",
            "checkpoint_latest.pt",
            "best_encoder.pt",
            "config.json",
            "metrics.json"
        )) {
            $source = Join-Path $runSource $name
            if (Test-Path -LiteralPath $source -PathType Leaf) {
                Copy-Item -LiteralPath $source -Destination $runDestination
            }
        }
    }

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path -LiteralPath $outputFile -PathType Leaf) {
        Remove-Item -LiteralPath $outputFile -Force
    }
    $archive = [System.IO.Compression.ZipFile]::Open(
        $outputFile,
        [System.IO.Compression.ZipArchiveMode]::Create
    )
    try {
        foreach ($file in Get-ChildItem -LiteralPath $staging -Recurse -File) {
            if (
                $file.Extension -eq ".pyc" -or
                $file.FullName -match "[\\/]__pycache__[\\/]"
            ) {
                continue
            }
            $relativePath = $file.FullName.Substring($staging.Length)
            $relativePath = $relativePath.TrimStart([char[]]"\/").Replace("\", "/")
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive,
                $file.FullName,
                $relativePath,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    } finally {
        $archive.Dispose()
    }

    $sizeMb = [math]::Round((Get-Item -LiteralPath $outputFile).Length / 1MB, 1)
    Write-Output "Created: $outputFile"
    Write-Output "Size: $sizeMb MB"
} finally {
    $stagingFull = [System.IO.Path]::GetFullPath($staging)
    if (-not $stagingFull.StartsWith($tempRoot)) {
        throw "Refusing to remove staging path outside temp: $stagingFull"
    }
    if (Test-Path -LiteralPath $stagingFull) {
        Remove-Item -LiteralPath $stagingFull -Recurse -Force
    }
}
