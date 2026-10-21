# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Position=0)]
    [string]$Command = '',
    [string]$Language = 'auto',
    [switch]$NoTts,
    [switch]$NoFastMode
)

# ============================================================
# BOOTSTRAP LOG — write to a guaranteed-writable location BEFORE
# any other logic. If run.ps1 fails before reaching the main log
# at $env:USERPROFILE\.openvino\log\, the bootstrap log catches
# the crash. Marvis-side debugging starts here:
#
#   $env:TEMP\camera-companion-bootstrap.log
#
# If THIS file is also missing, run.ps1 was never executed.
# ============================================================

$BootstrapLog = $null
foreach ($candidate in @(
    $env:TEMP,
    $env:TMP,
    'C:\Windows\Temp',
    [System.IO.Path]::GetTempPath()
)) {
    if ($candidate -and (Test-Path $candidate)) {
        $BootstrapLog = Join-Path $candidate 'camera-companion-bootstrap.log'
        break
    }
}

function Write-Bootstrap([string]$msg) {
    if (-not $BootstrapLog) { return }
    try {
        $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss.fff')
        Add-Content -Path $BootstrapLog -Value "[$stamp] $msg" -ErrorAction SilentlyContinue
    } catch { }
}

Write-Bootstrap "------------------------------------------------------------"
Write-Bootstrap "run.ps1 invoked"
Write-Bootstrap "  Command  = $Command"
Write-Bootstrap "  Language = $Language"
Write-Bootstrap "  NoTts    = $NoTts"
Write-Bootstrap "  NoFastMode = $NoFastMode"
Write-Bootstrap "  PSVersion= $($PSVersionTable.PSVersion)"
Write-Bootstrap "  PWD      = $PWD"
Write-Bootstrap "  PSScriptRoot = $PSScriptRoot"
Write-Bootstrap "  USERPROFILE  = $env:USERPROFILE"
Write-Bootstrap "  PATH (first 200 chars) = $($env:PATH.Substring(0, [Math]::Min($env:PATH.Length, 200)))"

# Marvis-side diagnostic line. Also goes to stderr so Marvis can
# show it in its UI even if the log file is unreachable.
$env:CAMERA_COMPANION_BOOTSTRAP_LOG = $BootstrapLog
Write-Host "[camera-companion] run.ps1 starting (bootstrap log: $BootstrapLog)"

$ErrorActionPreference = 'Stop'

# Trap any uncaught error so we always log the failure point.
trap {
    Write-Bootstrap "FATAL: $_"
    Write-Bootstrap "  ScriptStackTrace: $($_.ScriptStackTrace)"
    Write-Host "[camera-companion] run.ps1 FAILED: $_"
    exit 99
}

# --- Main log file under ~/.openvino/log/ ---
try {
    if ($env:USERPROFILE) {
        $LogDir = Join-Path $env:USERPROFILE '.openvino\log'
    } else {
        $LogDir = Join-Path ([System.IO.Path]::GetTempPath()) '.openvino-log'
        Write-Bootstrap "WARN: USERPROFILE unset; using temp log dir $LogDir"
    }
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        Write-Bootstrap "Created log dir: $LogDir"
    }
} catch {
    Write-Bootstrap "ERROR creating LogDir: $_"
    throw
}

$LogTimestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$LogFile = Join-Path $LogDir "camera-companion-client-$LogTimestamp.log"
try {
    Add-Content -Path $LogFile -Value "[$(Get-Date)] Log initialized."
    Write-Bootstrap "Main log file: $LogFile"
} catch {
    Write-Bootstrap "ERROR writing initial LogFile entry: $_"
    throw
}

function Write-Log([string]$msg) {
    try { Add-Content -Path $LogFile -Value "[$(Get-Date)] $msg" } catch {}
    Write-Bootstrap $msg
}

Write-Log "run.ps1 started: Command='$Command' Language='$Language' NoTts=$NoTts NoFastMode=$NoFastMode"

# --- Resolve paths early ---
$SkillRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Write-Log "SkillRoot = $SkillRoot"

if ($Command -eq '--help' -or $Command -eq '-h') {
    Write-Host 'Usage:'
    Write-Host '  scripts\run.ps1                       # start camera companion (auto language)'
    Write-Host '  scripts\run.ps1 -Language zh          # pin Chinese replies'
    Write-Host '  scripts\run.ps1 -Language en          # pin English replies'
    Write-Host '  scripts\run.ps1 -NoTts                # disable spoken replies'
    Write-Host '  scripts\run.ps1 -NoFastMode           # slow-path JSON COT envelope'
    Write-Host '  scripts\run.ps1 --continue            # resume first-run model download'
    Write-Host '  scripts\run.ps1 --status              # show server status + web URL'
    Write-Host '  scripts\run.ps1 --stop                # stop the server (free camera + GPU)'
    Write-Host '  scripts\run.ps1 --restart             # force-restart (reload code/models)'
    Write-Host '  scripts\run.ps1 --doctor              # run preflight diagnostics + exit'
    exit 0
}

# --- AIPC Check (with WMI fallback dev mode) ---
$PlatformScript = Join-Path $PSScriptRoot '..\bin\platform.ps1'
Write-Log "Resolved PLATFORM_SCRIPT=$PlatformScript"
if (-not (Test-Path $PlatformScript)) {
    Write-Log 'platform.ps1 not found; skipping AIPC check (dev mode).'
    Write-Host 'WARN: bin\platform.ps1 missing; skipping AIPC check.'
} else {
    $IsAipc = & $PlatformScript --is-aipc
    Write-Log "platform --is-aipc returned $IsAipc"
    if ($IsAipc -ne '1') {
        Write-Log 'This machine is not an Intel AIPC platform.'
        Write-Host 'This skill requires an Intel AIPC platform.'
        exit 1
    }
    Write-Host 'Intel AIPC platform detected. Continuing ...'
}

Push-Location $SkillRoot
Write-Log "Changed working directory to $SkillRoot"

# --- Read venv config from info.json ---
$EnvJson = Join-Path $SkillRoot 'info.json'
if (-not (Test-Path $EnvJson)) {
    Write-Host 'ERROR: info.json not found.'
    Write-Log "info.json missing at $EnvJson"
    Pop-Location
    exit 1
}
$Config = Get-Content $EnvJson -Raw | ConvertFrom-Json
$VenvName = $Config.venv_name
if (-not $VenvName) {
    Write-Host 'ERROR: venv_name not found in info.json.'
    Write-Log 'info.json missing venv_name'
    Pop-Location
    exit 1
}
$VenvDir = Join-Path $env:USERPROFILE ".openvino\venv\$VenvName"
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
Write-Log "Resolved VENV_PY=$VenvPy (exists: $(Test-Path $VenvPy))"

# --- --doctor: preflight diagnostics + exit ---
if ($Command -eq '--doctor') {
    Write-Log 'Running --doctor preflight diagnostics.'
    Write-Host '=== camera-companion --doctor ==='
    Write-Host "PSVersion        : $($PSVersionTable.PSVersion)"
    Write-Host "PWD              : $PWD"
    Write-Host "SkillRoot        : $SkillRoot"
    Write-Host "USERPROFILE      : $env:USERPROFILE"
    Write-Host "TEMP             : $env:TEMP"
    Write-Host "Bootstrap log    : $BootstrapLog"
    Write-Host "Main log         : $LogFile"
    Write-Host "Venv name        : $VenvName"
    Write-Host "Venv python      : $VenvPy"
    Write-Host "Venv python exists: $(Test-Path $VenvPy)"
    Write-Host "platform.ps1     : $PlatformScript ($(Test-Path $PlatformScript))"
    Write-Host "info.json        : $EnvJson ($(Test-Path $EnvJson))"
    $ModelDir = Join-Path $env:USERPROFILE '.openvino\models\MiniCPM-o-4_5-OV'
    Write-Host "Model dir        : $ModelDir ($(Test-Path $ModelDir))"
    $VadFile = Join-Path $env:USERPROFILE '.openvino\models\silero_vad\silero_vad.onnx'
    Write-Host "VAD model        : $VadFile ($(Test-Path $VadFile))"
    Write-Host "InstallEnv script: $(Join-Path $SkillRoot 'scripts\install-env.ps1') ($(Test-Path (Join-Path $SkillRoot 'scripts\install-env.ps1')))"
    Write-Host "Client script    : $(Join-Path $SkillRoot 'scripts\client.py') ($(Test-Path (Join-Path $SkillRoot 'scripts\client.py')))"
    Write-Host "Server script    : $(Join-Path $SkillRoot 'scripts\server.py') ($(Test-Path (Join-Path $SkillRoot 'scripts\server.py')))"
    if (Test-Path $VenvPy) {
        Write-Host '---- Python sys.version ----'
        & $VenvPy -c "import sys; print(sys.version)"
        Write-Host '---- Heavy imports (best-effort) ----'
        & $VenvPy -c "import importlib, traceback`nfor m in ('openvino', 'cv2', 'sounddevice', 'transformers', 'modelscope'):`n    try:`n        importlib.import_module(m)`n        print(f'OK  {m}')`n    except Exception as exc:`n        print(f'ERR {m}: {type(exc).__name__}: {exc}')"
    }
    Write-Host '=== end --doctor ==='
    Pop-Location
    exit 0
}

# --- Ensure environment (skip heavy setup on control-only ops) ---
if ($Command -eq '--status' -or $Command -eq '--server-status' -or
    $Command -eq '--stop' -or $Command -eq '--server-shutdown') {
    Write-Log "Control op '$Command'; skipping environment setup."
} else {
    $InstallEnv = Join-Path $SkillRoot 'scripts\install-env.ps1'
    if (Test-Path $InstallEnv) {
        Write-Host 'Setting up Python environment...'
        Write-Log "Calling install-env.ps1 at $InstallEnv"
        & $InstallEnv -SkillRoot $SkillRoot
        if ($LASTEXITCODE -ne 0) {
            Write-Log "install-env.ps1 exited $LASTEXITCODE; aborting."
            Pop-Location
            exit 1
        }
        Write-Log 'install-env.ps1 completed.'
    } else {
        Write-Host 'WARN: scripts\install-env.ps1 missing; assuming venv already set up.'
        Write-Log "install-env.ps1 missing at $InstallEnv"
        if (-not (Test-Path $VenvPy)) {
            Write-Host "ERROR: venv not found at $VenvPy and install-env.ps1 is missing."
            Write-Host '       Run build.ps1 to vendor install-env.ps1, or create the venv manually:'
            Write-Host "         python -m venv $VenvDir"
            Write-Host "         $VenvPy -m pip install -r requirements.txt"
            Write-Log "FATAL: no venv + no install-env.ps1"
            Pop-Location
            exit 1
        }
    }
}

Write-Host 'Launching controller...'
Write-Log "Dispatching to client.py: command='$Command'"

# --- Build the client arg list ---
$ClientArgs = @('scripts\client.py', '--language', $Language, '--log', $LogFile)
if ($NoTts) { $ClientArgs += '--no-tts' }
if ($NoFastMode) { $ClientArgs += '--no-fast-mode' }

# --- Launch client.py ---
switch ($Command) {
    '--continue' {
        & $VenvPy @ClientArgs --continue
    }
    { $_ -eq '--status' -or $_ -eq '--server-status' } {
        & $VenvPy scripts\client.py --server-status --log $LogFile
    }
    { $_ -eq '--stop' -or $_ -eq '--server-shutdown' } {
        # 1. Cooperative shutdown of the pipe-owning server.
        & $VenvPy scripts\client.py --server-shutdown --log $LogFile

        # 2. Backstop: the cooperative shutdown only stops the process
        #    bound to the pipe. Force-terminate ANY lingering
        #    camera-companion server.py process so the camera + mic +
        #    GPU are always released on the first --stop (camera_log
        #    session 20260702-131004 showed a duplicate lingering that
        #    had to be force-killed by hand). The match is scoped to this
        #    skill's server.py so unrelated Python is never touched.
        $pattern = '*local-scene-recognition*server.py*'
        function Get-ServerProcs {
            try {
                @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
                    Where-Object { $_.CommandLine -like $pattern })
            } catch {
                Write-Log "stop: Get-CimInstance failed: $_"
                @()
            }
        }
        $procs = Get-ServerProcs
        if ($procs.Count -gt 0) {
            $ids = ($procs | ForEach-Object { $_.ProcessId }) -join ','
            Write-Log "stop: lingering server PIDs [$ids]; waiting up to 5s for graceful exit"
            for ($i = 0; $i -lt 10 -and (Get-ServerProcs).Count -gt 0; $i++) {
                Start-Sleep -Milliseconds 500
            }
            foreach ($p in Get-ServerProcs) {
                Write-Log "stop: force-killing lingering server PID $($p.ProcessId)"
                try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop }
                catch { Write-Log "stop: Stop-Process $($p.ProcessId) failed: $_" }
            }
        } else {
            Write-Log "stop: no lingering server.py processes"
        }

        $remaining = (Get-ServerProcs).Count
        Write-Log "stop: complete; remaining server.py processes: $remaining"
        if ($remaining -gt 0) {
            Write-Host "停止未完全 / stop incomplete: $remaining process(es) still running."
        } else {
            Write-Host "已停止, 资源已释放 / stopped; camera + mic + GPU released."
        }
        Pop-Location
        exit ([int]($remaining -gt 0))
    }
    '--restart' {
        & $VenvPy @ClientArgs --restart
    }
    default {
        & $VenvPy @ClientArgs
    }
}
$exitCode = $LASTEXITCODE
Write-Log "client.py exited with code $exitCode"
Write-Bootstrap "client.py exit code = $exitCode"
Pop-Location
exit $exitCode
