$ErrorActionPreference = 'Stop'

Push-Location $PSScriptRoot
try {
    $usage = 'Usage: platform.ps1 --is-aipc [--test <cpu-model-name>]'
    $testName = $null
    if ($args.Count -eq 3 -and $args[0] -ceq '--is-aipc' -and $args[1] -ceq '--test') {
        $testName = ([string]$args[2]).Trim()
    }
    elseif ($args.Count -ne 1 -or $args[0] -cne '--is-aipc') {
        [Console]::Error.WriteLine($usage)
        exit 2
    }

    if ($null -ne $testName) {
        $modelName = $testName
    }
    else {
        try {
            $modelName = Get-CimInstance -Query 'SELECT Name FROM Win32_Processor' |
                ForEach-Object { ([string]$_.Name).Trim() } |
                Where-Object { $_ } |
                Select-Object -First 1
        }
        catch {
            [Console]::Error.WriteLine('error: failed to query WMI')
            exit 1
        }
    }

    $isAipc = $false
    if ($modelName) {
        # Drop trailing letters-only suffix tokens such as "Plus" or "BFC".
        $tokens = @($modelName -split ' ' | Where-Object { $_ })
        while ($tokens.Count -gt 1 -and $tokens[-1] -cmatch '^[A-Za-z]+$') {
            $tokens = $tokens[0..($tokens.Count - 2)]
        }
        $code = if ($tokens.Count -gt 0) { $tokens[-1] } else { '' }

        if ($code.Length -gt 0) {
            $firstCharacter = [int][char]$code[0]
            $startsWithDigit = $firstCharacter -ge [int][char]'0' -and
                $firstCharacter -le [int][char]'9'

            if ($startsWithDigit -and $code[0] -ne [char]'1') {
                $isAipc = $modelName.Contains('Ultra') -or
                    $firstCharacter -ge [int][char]'3'
            }
        }
    }

    if ($isAipc) {
        Write-Output '1'
    }
    else {
        Write-Output '0'
    }
}
finally {
    Pop-Location
}
