# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

param(
    [string]$SkillRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

$SharedUvDir = Join-Path $env:USERPROFILE '.local\bin'
$UvExe = Join-Path $SharedUvDir 'uv.exe'
$LegacyUvExe = Join-Path $SkillRoot 'bin\uv.exe'
$EnvJson = Join-Path $SkillRoot 'info.json'
#$MirrorIndex = 'https://mirrors.cloud.tencent.com/pypi/simple/'
$MirrorIndex = 'http://mirrors.aliyun.com/pypi/simple/'

if (-not (Test-Path $EnvJson)) {
    Write-Host "ERROR: info.json not found at `"$EnvJson`"."
    exit 1
}

$config = Get-Content $EnvJson -Raw | ConvertFrom-Json
$VenvName = $config.venv_name
$PythonVersion = if ($config.python_version) { $config.python_version } else { '3.11' }

$VenvDir = Join-Path $env:USERPROFILE ".openvino\venv\$VenvName"
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'

# Restricted hosts can reject uv's normal %LOCALAPPDATA% cache cleanup. Keep
# the cache in the OS-managed temporary directory for every skill instead.
$env:UV_CACHE_DIR = Join-Path ([System.IO.Path]::GetTempPath()) 'openvino-uv-cache'

# --- Logging ---
$LogDir = Join-Path $env:USERPROFILE '.openvino\log'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$LogTimestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$LogFile = Join-Path $LogDir "install-env-$LogTimestamp.log"
Add-Content $LogFile "[$(Get-Date)] Log initialized for $SkillRoot."

function Write-Log($msg) { Add-Content $LogFile "[$(Get-Date)] $msg" }

function Get-WheelPackageInfo {
    param(
        [System.IO.FileInfo]$WheelFile
    )

    $wheelBaseName = [System.IO.Path]::GetFileNameWithoutExtension($WheelFile.Name)
    $wheelParts = $wheelBaseName -split '-'
    if ($wheelParts.Length -lt 2) {
        return $null
    }

    [PSCustomObject]@{
        PackageName = ($wheelParts[0] -replace '[-_.]+', '-').ToLowerInvariant()
        Version     = $wheelParts[1]
    }
}

function Invoke-PythonScript {
    param(
        [string]$PythonExe,
        [string]$Script,
        [string[]]$Arguments = @(),
        [switch]$SuppressStderr
    )

    $tempScript = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), ([System.IO.Path]::GetRandomFileName() + '.py'))
    try {
        Set-Content -Path $tempScript -Value $Script -Encoding ascii
        if ($SuppressStderr) {
            return & $PythonExe $tempScript @Arguments 2>$null
        }

        return & $PythonExe $tempScript @Arguments
    } finally {
        Remove-Item $tempScript -Force -ErrorAction SilentlyContinue
    }
}

function Get-InstalledPackageVersion {
    param(
        [string]$PythonExe,
        [string]$PackageName
    )

    $script = @'
import importlib.metadata
import re
import sys

def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()

target = normalize(sys.argv[1])
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if name and normalize(name) == target:
        print(distribution.version)
        sys.exit(0)

sys.exit(1)
'@

    $installedVersion = Invoke-PythonScript -PythonExe $PythonExe -Script $script -Arguments @($PackageName) -SuppressStderr
    if ($LASTEXITCODE -ne 0 -or -not $installedVersion) {
        return $null
    }

    return ($installedVersion | Select-Object -First 1).Trim()
}

function Test-InstalledVersionOlderThanWheel {
    param(
        [string]$PythonExe,
        [string]$InstalledVersion,
        [string]$WheelVersion
    )

    $script = @'
import re
import sys

installed, wheel = sys.argv[1], sys.argv[2]

def split_version(value):
    parts = []
    for part in re.split(r"([0-9]+)", value.lower()):
        if not part:
            continue
        parts.append(int(part) if part.isdigit() else part)
    return parts

version_class = None
try:
    from packaging.version import Version
    version_class = Version
except Exception:
    try:
        from pip._vendor.packaging.version import Version
        version_class = Version
    except Exception:
        version_class = None

if version_class is not None:
    try:
        sys.exit(0 if version_class(installed) < version_class(wheel) else 1)
    except Exception:
        pass

sys.exit(0 if split_version(installed) < split_version(wheel) else 1)
'@

    Invoke-PythonScript -PythonExe $PythonExe -Script $script -Arguments @($InstalledVersion, $WheelVersion) | Out-Null
    return $LASTEXITCODE -eq 0
}

# --- uv bootstrap helpers ---
# uv is never fetched from a third-party file host. Each helper installs uv
# through an official channel and returns the path of the uv.exe it produced,
# or $null when that channel is unavailable. None of them put the new uv.exe on
# this process's PATH, so every helper resolves the file by path itself.

function Resolve-WingetUv {
    # winget's portable installer publishes a symlink under ...\WinGet\Links.
    # Prefer the real file under ...\WinGet\Packages so the shared copy made by
    # the caller is a plain file: restricted hosts such as WorkBuddy reject
    # symlinks.
    if (-not $env:LOCALAPPDATA) { return $null }
    foreach ($relative in @('Microsoft\WinGet\Packages', 'Microsoft\WinGet\Links')) {
        $root = Join-Path $env:LOCALAPPDATA $relative
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $hit = Get-ChildItem -LiteralPath $root -Filter 'uv.exe' -Recurse -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

function Install-UvWithPip {
    # `pip install --user` keeps the host's system site-packages untouched and
    # works without administrator rights. mirror first, PyPI second --
    # same order as every other pip install in this script.
    $python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    if ($python) {
        $runner = $python
        $prefix = @('-m', 'pip')
    } else {
        $runner = (Get-Command pip.exe -ErrorAction SilentlyContinue).Source
        if (-not $runner) { $runner = (Get-Command pip3.exe -ErrorAction SilentlyContinue).Source }
        if (-not $runner) { return $null }
        $prefix = @()
    }

    Write-Host '  Installing uv with pip...'
    Write-Log "Installing uv via pip: $runner"

    # pip refuses `--user` outright inside a virtual environment ("User
    # site-packages are not visible in this virtualenv"), and the python on a
    # host's PATH very often IS a venv. Pick the flag accordingly, and still
    # retry without it when pip.exe was found without a python to interrogate.
    $flagSets = New-Object System.Collections.ArrayList
    $inVirtualEnv = $false
    if ($python) {
        & $python -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $inVirtualEnv = $true
            Write-Log 'pip belongs to a virtual environment; installing without --user.'
        }
    }
    if (-not $inVirtualEnv) { [void]$flagSets.Add(@('--user')) }
    [void]$flagSets.Add(@())

    $installed = $false
    foreach ($index in @($MirrorIndex, '')) {
        foreach ($flags in $flagSets) {
            $pipArguments = $prefix + @('install') + $flags + @('--disable-pip-version-check', 'uv')
            if ($index) { $pipArguments += @('--index-url', $index) }
            # Out-Host keeps the tool's progress visible without letting it
            # become part of this function's return value: a PowerShell
            # function returns everything written to the output stream.
            & $runner @pipArguments | Out-Host
            if ($LASTEXITCODE -eq 0) { $installed = $true; break }
        }
        if ($installed) { break }
    }
    if (-not $installed) {
        Write-Log 'pip could not install uv.'
        return $null
    }

    # The scripts directory pip installed into is not on PATH; ask Python for
    # both the per-user and the current-environment locations.
    $scriptDirs = @()
    if ($python) {
        foreach ($scheme in @('nt_user', '')) {
            $expression = if ($scheme) {
                "import sysconfig; print(sysconfig.get_path('scripts', scheme='$scheme'))"
            } else {
                "import sysconfig; print(sysconfig.get_path('scripts'))"
            }
            $reported = & $python -c $expression
            if ($LASTEXITCODE -eq 0 -and $reported) { $scriptDirs += ($reported | Select-Object -First 1).Trim() }
        }
    } elseif ($runner) {
        $scriptDirs += (Split-Path $runner -Parent)
    }
    if ($env:APPDATA) {
        $scriptDirs += (Get-ChildItem -LiteralPath (Join-Path $env:APPDATA 'Python') -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Join-Path $_.FullName 'Scripts' })
    }
    foreach ($dir in $scriptDirs) {
        if (-not $dir) { continue }
        $candidate = Join-Path $dir 'uv.exe'
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }

    Write-Log 'pip reported success but uv.exe was not found in any scripts directory.'
    return $null
}

function Install-UvWithWinget {
    $winget = (Get-Command winget.exe -ErrorAction SilentlyContinue).Source
    if (-not $winget) { return $null }

    Write-Host '  Installing uv with winget...'
    Write-Log 'Installing uv via winget (astral-sh.uv).'
    # The agreement flags are required for an unattended run; without them
    # winget waits for interactive confirmation and the skill appears to hang.
    & $winget install --id=astral-sh.uv -e --silent `
        --accept-source-agreements --accept-package-agreements | Out-Host
    Write-Log "winget exit code: $LASTEXITCODE"
    return (Resolve-WingetUv)
}

function Install-UvFromArchive {
    # Last resort: the pinned release archive published by Astral itself.
    $url = 'https://releases.astral.sh/github/uv/releases/download/0.12.14/uv-x86_64-pc-windows-msvc.zip'
    $expectedHash = '02cecb254f37f3396da625680796d3990164a590fb5611d46f719ad83e7b9a89'
    Write-Host '  Downloading the uv release archive...'
    Write-Log "Downloading uv archive from $url"

    $archive = Join-Path ([System.IO.Path]::GetTempPath()) "uv-archive-$PID.zip"
    $extractDir = Join-Path ([System.IO.Path]::GetTempPath()) "uv-archive-$PID"
    Remove-Item -LiteralPath $archive, $extractDir -Recurse -Force -ErrorAction SilentlyContinue

    $curlExe = (Get-Command curl.exe -ErrorAction SilentlyContinue).Source
    if ($curlExe) {
        & $curlExe -fL --ssl-no-revoke -o $archive $url | Out-Host
    }
    if (-not (Test-Path -LiteralPath $archive)) {
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $archive
        } catch {
            Write-Log "uv archive download failed: $($_.Exception.Message)"
        }
    }
    if (-not (Test-Path -LiteralPath $archive)) { return $null }

    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        Write-Log "uv archive SHA-256 mismatch: expected $expectedHash, got $actualHash"
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
        return $null
    }

    try {
        Expand-Archive -LiteralPath $archive -DestinationPath $extractDir -Force
    } catch {
        Write-Log "uv archive could not be extracted: $($_.Exception.Message)"
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
        return $null
    }
    Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue

    $hit = Get-ChildItem -LiteralPath $extractDir -Filter 'uv.exe' -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($hit) { return $hit.FullName }

    Write-Log 'uv.exe was not present in the release archive.'
    return $null
}

# --- Step 1: Ensure UV ---
# One shared private copy lives at %USERPROFILE%\.local\bin\uv.exe. Sources are
# tried in order and the first usable one wins:
#   1. an existing shared copy
#   2. a uv.exe bundled with an older skill package (bin\uv.exe)
#   3. a uv.exe already on PATH
#   4. pip install --user uv
#   5. winget install --id=astral-sh.uv -e
#   6. the pinned release archive at https://releases.astral.sh
# Whatever a step produces is copied into the shared location as a plain file.
Write-Host '[1/4] Checking UV installation...'

# pip, winget, and uv itself all write normal progress to
# stderr. Under $ErrorActionPreference = 'Stop' that becomes a terminating
# NativeCommandError whenever the caller captures this script through a `2>&1 |`
# pipeline (run.ps1's test harness, and hosts that capture run.ps1's combined
# output, do exactly this), which would abort the install mid-way and leave a
# broken venv. Every step from here on checks its result explicitly, so switch
# to 'Continue' and restore the previous preference after the uv section.
$PrevErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'

if (-not (Test-Path $UvExe)) {
    if (-not (Test-Path $SharedUvDir)) {
        New-Item -ItemType Directory -Path $SharedUvDir -Force | Out-Null
    }

    $uvSource = $null
    if (Test-Path $LegacyUvExe) {
        Write-Host "  Migrating uv.exe from `"$LegacyUvExe`" to `"$UvExe`"..."
        $uvSource = $LegacyUvExe
    } else {
        $uvSource = (Get-Command uv.exe -ErrorAction SilentlyContinue).Source
        if ($uvSource) {
            Write-Host "  Found uv.exe on PATH at `"$uvSource`". Copying to `"$UvExe`"..."
        }
    }

    foreach ($method in @('pip', 'winget', 'archive')) {
        if ($uvSource -or (Test-Path $UvExe)) { break }
        switch ($method) {
            'pip'       { $uvSource = Install-UvWithPip }
            'winget'    { $uvSource = Install-UvWithWinget }
            'archive'   { $uvSource = Install-UvFromArchive }
        }
    }

    if (-not (Test-Path $UvExe)) {
        if (-not $uvSource) {
            Write-Host 'ERROR: Failed to install uv.exe, please check your network connection and try again.'
            Write-Log 'Every uv bootstrap channel failed.'
            $ErrorActionPreference = $PrevErrorActionPreference
            exit 1
        }

        # Copy through a private temporary file, then rename it into the shared
        # location. Two skills can be launched at once on a fresh host; no
        # caller may ever observe a partially copied uv.exe.
        $UvTemp = Join-Path $SharedUvDir "uv.exe.partial-$PID"
        try {
            Copy-Item -LiteralPath $uvSource -Destination $UvTemp -Force
            if (-not (Test-Path -LiteralPath $UvTemp)) {
                Write-Host 'ERROR: Failed to install uv.exe, please check your network connection and try again.'
                Write-Log "Could not copy uv.exe from $uvSource."
                $ErrorActionPreference = $PrevErrorActionPreference
                exit 1
            }
            Move-Item -LiteralPath $UvTemp -Destination $UvExe -Force
        } finally {
            Remove-Item -LiteralPath $UvTemp -Force -ErrorAction SilentlyContinue
        }
    }
}
Write-Host "  UV is available: $UvExe"

# --- Step 2: Create Python venv ---
Write-Host "[2/4] Ensuring Python venv at $VenvDir ..."
if (-not (Test-Path $VenvPy)) {
    Write-Host "  Creating Python $PythonVersion virtual environment..."
    # --seed installs pip/setuptools/wheel into the venv. Required because
    # setup_extra.py installs MeloTTS via `python -m pip install` (uv venv
    # creates a pip-less environment by default, which fails with
    # "No module named pip").
    # --clear replaces any pre-existing directory. Without it, a venv creation
    # that was interrupted mid-way (leaving a dir with only .gitignore /
    # CACHEDIR.TAG and no python.exe) makes every later run fail with
    # "A directory already exists" — uv refuses to overwrite by default. The
    # outer `if (-not (Test-Path $VenvPy))` guard means we only get here when
    # the venv is missing or incomplete, so clearing is always safe.
    & $UvExe venv --clear --seed --python $PythonVersion $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'ERROR: Failed to create Python virtual environment.'
        exit 1
    }
}

# Self-heal: ensure pip is present in the venv even when we DID NOT just create
# it. The --seed flag above only runs on the create branch, so a venv built by
# an OLDER install-env.ps1 (before --seed was added) is reused as-is and stays
# pip-less. setup_extra.py then fails every `python -m pip install` with
# "No module named pip" and MeloTTS/TTS silently never installs. Probe pip on
# every run and seed it on demand so pre-existing pip-less venvs recover without
# a full rebuild. Cheap when pip is already there (one --version call).
& $VenvPy -m pip --version > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host '  pip missing from venv (older uv venv had no --seed); installing pip...'
    Write-Log 'pip missing from venv; seeding pip via uv pip install.'
    & $UvExe pip install --python $VenvPy pip --index-url $MirrorIndex
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Mirror($MirrorIndex) failed, retrying pip install from official PyPI..."
        & $UvExe pip install --python $VenvPy pip
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'ERROR: Failed to install pip into the venv.'
        Write-Log 'ERROR: Failed to seed pip into the venv.'
        exit 1
    }
}
Write-Host "  Python venv is ready."

# --- Step 3: Install requirements.txt ---
Write-Host '[3/4] Installing requirements...'
$RequirementsFile = Join-Path $SkillRoot 'requirements.txt'
$WheelsDir = Join-Path $SkillRoot 'wheels'
Write-Log "Installing requirements from $RequirementsFile with wheels from $WheelsDir"

if (Test-Path $RequirementsFile) {
    $RequirementsShaFile = Join-Path $VenvDir 'requirements.sha'
    $RequirementsHash = (Get-FileHash -Path $RequirementsFile -Algorithm SHA256).Hash
    $ExistingRequirementsHash = if (Test-Path $RequirementsShaFile) { (Get-Content $RequirementsShaFile -Raw).Trim() } else { '' }

    if ($ExistingRequirementsHash -eq $RequirementsHash) {
        Write-Host '  requirements.txt unchanged. Skipping install.'
        Write-Log 'requirements.txt unchanged. Skipping install.'
    } else {
        $basePipArgs = @('pip', 'install', '--python', $VenvPy, '-r', $RequirementsFile)
        if (Test-Path $WheelsDir) {
            $basePipArgs += '--find-links'
            $basePipArgs += $WheelsDir
        }

        Write-Host "  Installing from requirements.txt (Mirror($MirrorIndex))..."
        Write-Log "Installing from requirements.txt with args: $($basePipArgs -join ' ') --index-url $MirrorIndex"
        & $UvExe @basePipArgs --index-url $MirrorIndex
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Mirror($MirrorIndex) failed, retrying from official PyPI..."
            Write-Log "  Mirror($MirrorIndex) failed, retrying from official PyPI..."
            & $UvExe @basePipArgs
            if ($LASTEXITCODE -ne 0) {
                Write-Host 'ERROR: Failed to install requirements.'
                Write-Log 'ERROR: Failed to install requirements.'
                exit 1
            }
        }

        Set-Content -Path $RequirementsShaFile -Value $RequirementsHash -Encoding ascii
    }
} else {
    Write-Host '  No requirements.txt found. Skipping.'
}
Write-Log "Finished installing requirements."

# --- Step 4: Install extra .whl files ---
Write-Host '[4/4] Installing extra wheel files...'
$WheelUrlsFile = Join-Path $SkillRoot 'wheel-urls.txt'
if (Test-Path $WheelUrlsFile) {
    $curlExe = (Get-Command curl.exe -ErrorAction SilentlyContinue).Source
    if (-not $curlExe) {
        Write-Host 'ERROR: curl.exe is required to download the special wheel.'
        exit 1
    }

    if (-not (Test-Path $WheelsDir)) {
        New-Item -ItemType Directory -Path $WheelsDir -Force | Out-Null
    }

    foreach ($wheelUrlLine in Get-Content -LiteralPath $WheelUrlsFile) {
        $wheelEntry = $wheelUrlLine.Trim()
        if (-not $wheelEntry -or $wheelEntry.StartsWith('#')) { continue }

        $wheelFields = $wheelEntry -split '\s+', 2
        if ($wheelFields.Count -ne 2 -or $wheelFields[1] -notmatch '^sha256:[A-Fa-f0-9]{64}$') {
            Write-Host "ERROR: Invalid wheel entry in $WheelUrlsFile`: $wheelEntry"
            Write-Log "ERROR: Invalid wheel entry: $wheelEntry"
            exit 1
        }
        $wheelUrl = $wheelFields[0]
        $expectedWheelHash = $wheelFields[1].Substring('sha256:'.Length)

        try {
            $wheelUri = [System.Uri]$wheelUrl
            if ($wheelUri.Scheme -notin @('http', 'https')) { throw 'Only HTTP(S) wheel URLs are supported.' }
            $wheelName = [System.Uri]::UnescapeDataString([System.IO.Path]::GetFileName($wheelUri.AbsolutePath))
            if (-not $wheelName.EndsWith('.whl', [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'The URL does not point to a .whl file.'
            }
        } catch {
            Write-Host "ERROR: Invalid wheel URL in $WheelUrlsFile`: $wheelUrl"
            Write-Log "ERROR: Invalid wheel URL: $wheelUrl ($($_.Exception.Message))"
            exit 1
        }

        $wheelPath = Join-Path $WheelsDir $wheelName
        if (-not (Test-Path $wheelPath)) {
            $partialPath = "$wheelPath.partial-$PID"
            Write-Host "  Downloading $wheelName with curl..."
            Write-Log "Downloading managed wheel with curl GET: $wheelUrl"
            try {
                & $curlExe -fL --ssl-no-revoke --retry 3 --retry-delay 1 -o $partialPath $wheelUrl
                if ($LASTEXITCODE -ne 0 -or -not (Test-Path $partialPath) -or (Get-Item $partialPath).Length -eq 0) {
                    Write-Host "ERROR: Failed to download $wheelName."
                    Write-Log "ERROR: curl failed to download $wheelUrl (exit=$LASTEXITCODE)."
                    exit 1
                }
                Move-Item -LiteralPath $partialPath -Destination $wheelPath -Force
            } finally {
                Remove-Item -LiteralPath $partialPath -Force -ErrorAction SilentlyContinue
            }
        }

        $managedWheel = Get-Item -LiteralPath $wheelPath
        $wheelInfo = Get-WheelPackageInfo -WheelFile $managedWheel
        if (-not $wheelInfo) {
            Write-Host "ERROR: Unable to parse package info from $wheelName."
            exit 1
        }

        $wheelHash = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash
        if (-not $wheelHash.Equals($expectedWheelHash, [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $wheelPath -Force -ErrorAction SilentlyContinue
            Write-Host "ERROR: SHA-256 mismatch for $wheelName."
            Write-Log "ERROR: SHA-256 mismatch for ${wheelName}: expected $expectedWheelHash, got $wheelHash."
            exit 1
        }
        $wheelShaFile = Join-Path $VenvDir "managed-wheel-$($wheelInfo.PackageName).sha"
        $installedVersion = Get-InstalledPackageVersion -PythonExe $VenvPy -PackageName $wheelInfo.PackageName
        $installedWheelHash = if (Test-Path $wheelShaFile) { (Get-Content $wheelShaFile -Raw).Trim() } else { '' }
        if ($installedVersion -and $installedWheelHash -eq $wheelHash) {
            Write-Host "  $($wheelInfo.PackageName) managed wheel is already installed. Skipping."
            continue
        }

        # The special build has the same version as the public package. A local
        # path plus --reinstall-package replaces the public build
        $managedPipArgs = @(
            'pip', 'install', '--python', $VenvPy, $wheelPath,
            '--reinstall-package', $wheelInfo.PackageName,
            '--find-links', $WheelsDir
        )
        Write-Host "  Installing managed wheel $wheelName..."
        & $UvExe @managedPipArgs --index-url $MirrorIndex
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Mirror($MirrorIndex) failed for $wheelName, retrying from official PyPI..."
            & $UvExe @managedPipArgs
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Failed to install $wheelName."
                exit 1
            }
        }
        Set-Content -Path $wheelShaFile -Value $wheelHash -Encoding ascii
    }
}

if (Test-Path $WheelsDir) {
    $whlFiles = Get-ChildItem (Join-Path $WheelsDir '*.whl') -ErrorAction SilentlyContinue
    if ($whlFiles) {
        foreach ($whl in $whlFiles) {
            $wheelInfo = Get-WheelPackageInfo -WheelFile $whl
            $shouldInstallWheel = $true

            if ($wheelInfo) {
                $installedVersion = Get-InstalledPackageVersion -PythonExe $VenvPy -PackageName $wheelInfo.PackageName
                if (-not $installedVersion) {
                    Write-Host "  $($wheelInfo.PackageName) is not installed. Installing $($whl.Name)..."
                } elseif (Test-InstalledVersionOlderThanWheel -PythonExe $VenvPy -InstalledVersion $installedVersion -WheelVersion $wheelInfo.Version) {
                    Write-Host "  $($wheelInfo.PackageName) $installedVersion is older than wheel $($wheelInfo.Version). Installing $($whl.Name)..."
                } else {
                    Write-Host "  $($wheelInfo.PackageName) $installedVersion is already newer than or equal to wheel $($wheelInfo.Version). Skipping."
                    $shouldInstallWheel = $false
                }
            } else {
                Write-Host "  Unable to parse package info from $($whl.Name). Installing."
            }

            if (-not $shouldInstallWheel) {
                continue
            }

            Write-Host "  Installing $($whl.Name) (Mirror($MirrorIndex) for deps)..."
            & $UvExe pip install --python $VenvPy $whl.FullName --index-url $MirrorIndex --find-links $WheelsDir
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  Mirror($MirrorIndex) failed for $($whl.Name), retrying from official PyPI..."
                & $UvExe pip install --python $VenvPy $whl.FullName --find-links $WheelsDir
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "ERROR: Failed to install $($whl.Name)."
                    exit 1
                }
            }
        }
    } else {
        Write-Host '  No wheel files found. Skipping.'
    }
} else {
    Write-Host '  No wheels directory found. Skipping.'
}

# Restore the caller's error handling now that the uv (stderr-noisy) steps are done.
$ErrorActionPreference = $PrevErrorActionPreference

Write-Host ''
Write-Host 'Environment setup complete.'
exit 0
