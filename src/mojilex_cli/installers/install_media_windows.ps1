[CmdletBinding()]
param(
    [switch]$InstallTgs,
    [switch]$InstallWebm
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$rlottieRevision = '683bbaa39dd0d366cf6b4bc300b4dfbee677ea6b'

function Require-Command {
    param([Parameter(Mandatory = $true)][string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "$Name was not found on PATH."
    }
    return $command.Source
}

$winget = Get-Command 'winget.exe' -ErrorAction SilentlyContinue

if ($InstallWebm -and $null -eq (Get-Command 'ffmpeg.exe' -ErrorAction SilentlyContinue)) {
    if ($null -eq $winget) {
        throw 'FFmpeg is missing and winget is unavailable. Install FFmpeg and rerun this command.'
    }
    Write-Host 'Installing FFmpeg...'
    & $winget.Source install --id Gyan.FFmpeg --exact `
        --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg installation failed with exit code $LASTEXITCODE."
    }
}

if ($InstallTgs) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $installation = ''
    if (Test-Path -LiteralPath $vswhere) {
        $installation = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    if ([string]::IsNullOrWhiteSpace($installation)) {
        if ($null -eq $winget) {
            throw 'Visual C++ Build Tools are missing and winget is unavailable. Install Microsoft Visual Studio 2022 Build Tools with the Desktop development with C++ workload, then rerun this command.'
        }
        Write-Host 'Installing Microsoft Visual Studio 2022 Build Tools (C++ workload)...'
        & $winget.Source install --id Microsoft.VisualStudio.2022.BuildTools --exact `
            --accept-package-agreements --accept-source-agreements --force `
            --override '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
        if ($LASTEXITCODE -ne 0) {
            throw "Visual Studio Build Tools installation failed with exit code $LASTEXITCODE."
        }
    }

    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw 'Visual Studio Installer did not provide vswhere.exe after installation.'
    }

    $installation = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installation)) {
        throw 'Visual C++ x64 build tools are unavailable.'
    }

    $devShellModule = Join-Path $installation 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    Import-Module $devShellModule
    Enter-VsDevShell -VsInstallPath $installation -SkipAutomaticLocation `
        -DevCmdArguments '-arch=x64 -host_arch=x64'

    $git = Require-Command 'git.exe'
    $uv = Require-Command 'uv.exe'

    $adapterSource = Join-Path $PSScriptRoot 'rlottie_rgba_renderer.cpp'
    if (-not (Test-Path -LiteralPath $adapterSource)) {
        $sourceCheckoutAdapter = Join-Path $PSScriptRoot '..\..\..\tools\rlottie_rgba_renderer.cpp'
        if (Test-Path -LiteralPath $sourceCheckoutAdapter) {
            $adapterSource = (Resolve-Path -LiteralPath $sourceCheckoutAdapter).Path
        } else {
            throw 'The bundled MojiLex rlottie adapter source is missing.'
        }
    }

    $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) `
        ('mojilex-rlottie-install-' + [guid]::NewGuid().ToString('N'))
    $source = Join-Path $temporaryRoot 'rlottie'
    $build = Join-Path $temporaryRoot 'rlottie-build'
    $buildEnvironment = Join-Path $temporaryRoot 'build-environment'
    $adapterObject = Join-Path $temporaryRoot 'rlottie_rgba_renderer.obj'
    $destinationDirectory = Join-Path $env:USERPROFILE '.local\bin'
    $destination = Join-Path $destinationDirectory 'mojilex-rlottie-rgba.exe'

    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    try {
    Write-Host "Building pinned rlottie revision $rlottieRevision..."
    & $git clone --no-checkout --filter=blob:none https://github.com/Samsung/rlottie.git $source
    if ($LASTEXITCODE -ne 0) {
        throw "rlottie clone failed with exit code $LASTEXITCODE."
    }
    & $git -C $source checkout $rlottieRevision
    if ($LASTEXITCODE -ne 0) {
        throw "rlottie checkout failed with exit code $LASTEXITCODE."
    }

    & $uv venv $buildEnvironment --python 3.11
    if ($LASTEXITCODE -ne 0) {
        throw "Build environment creation failed with exit code $LASTEXITCODE."
    }
    $buildPython = Join-Path $buildEnvironment 'Scripts\python.exe'
    & $uv pip install --python $buildPython 'meson>=1.3,<2' 'ninja>=1.11,<2'
    if ($LASTEXITCODE -ne 0) {
        throw "Meson/Ninja installation failed with exit code $LASTEXITCODE."
    }
    $env:PATH = (Join-Path $buildEnvironment 'Scripts') + ';' + $env:PATH
    $meson = Join-Path $buildEnvironment 'Scripts\meson.exe'

    & $meson setup $build $source -Dexample=false -Dtest=false -Dmodule=false `
        -Ddefault_library=static --buildtype=release
    if ($LASTEXITCODE -ne 0) {
        throw "rlottie configuration failed with exit code $LASTEXITCODE."
    }
    & $meson compile -C $build
    if ($LASTEXITCODE -ne 0) {
        throw "rlottie build failed with exit code $LASTEXITCODE."
    }

    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    $rlottieLibrary = Join-Path $build 'src\librlottie.a'
    & cl.exe /nologo /std:c++14 /EHsc /O2 /MD /W4 /WX "/I$source\inc" `
        "/Fo$adapterObject" $adapterSource /link $rlottieLibrary Shlwapi.lib "/OUT:$destination"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $destination)) {
        throw "MojiLex TGS adapter build failed with exit code $LASTEXITCODE."
    }
    } finally {
    $resolvedTemporaryRoot = [IO.Path]::GetFullPath($temporaryRoot)
    $systemTemporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if (
        $resolvedTemporaryRoot.StartsWith($systemTemporaryRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedTemporaryRoot).StartsWith('mojilex-rlottie-install-')
    ) {
        Remove-Item -LiteralPath $resolvedTemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    }

    Write-Host "Installed: $destination"
}
