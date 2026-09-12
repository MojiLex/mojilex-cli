[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$ParentProcessId,
    [Parameter(Mandatory = $true)][string]$UvPath,
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [Parameter(Mandatory = $true)][string]$AdapterPath,
    [switch]$KeepData
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

try {
    while ($null -ne (Get-Process -Id $ParentProcessId -ErrorAction SilentlyContinue)) {
        Start-Sleep -Milliseconds 200
    }

    & $UvPath tool uninstall mojilex-cli
    if ($LASTEXITCODE -ne 0) {
        throw "uv tool uninstall failed with exit code $LASTEXITCODE."
    }

    if (Test-Path -LiteralPath $AdapterPath -PathType Leaf) {
        Remove-Item -LiteralPath $AdapterPath -Force
    }
    if (-not $KeepData -and (Test-Path -LiteralPath $DataRoot -PathType Container)) {
        $resolvedDataRoot = [IO.Path]::GetFullPath($DataRoot)
        $localAppDataRoot = [IO.Path]::GetFullPath($env:LOCALAPPDATA)
        if (
            -not $resolvedDataRoot.StartsWith($localAppDataRoot, [StringComparison]::OrdinalIgnoreCase) -or
            (Split-Path -Leaf $resolvedDataRoot) -ne 'mojilex'
        ) {
            throw "Refusing to remove unexpected data directory: $resolvedDataRoot"
        }
        $dataItem = Get-Item -LiteralPath $resolvedDataRoot -Force
        if (($dataItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing to recursively remove a reparse point: $resolvedDataRoot"
        }
        Remove-Item -LiteralPath $resolvedDataRoot -Recurse -Force
    }
} finally {
    Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue
}
