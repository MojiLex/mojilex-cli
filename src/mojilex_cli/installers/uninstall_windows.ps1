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

    if (-not $KeepData) {
        # python-keyring's Windows backend stores these generic credentials under
        # the service name and, after collisions, username@service. Remove them
        # only once uv has confirmed that the package was uninstalled.
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class MojiLexCredentialApi {
    [DllImport("advapi32.dll", EntryPoint = "CredDeleteW", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool Delete(string target, int type, int flags);
}
'@
        $targets = @('mojilex-cli')
        foreach ($name in @('TELEGRAM_BOT_TOKEN', 'GEMINI_API_KEY', 'OPENAI_API_KEY')) {
            $targets += "${name}@mojilex-cli"
        }
        foreach ($target in $targets) {
            if (-not [MojiLexCredentialApi]::Delete($target, 1, 0)) {
                $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
                if ($errorCode -ne 1168) {
                    throw "Windows Credential Manager could not remove MojiLex credentials (error $errorCode)."
                }
            }
        }
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
