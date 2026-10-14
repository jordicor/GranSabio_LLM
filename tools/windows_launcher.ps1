param(
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [ValidateRange(0, 65535)][int]$Port = 0
)

$ErrorActionPreference = 'Stop'
$script:GranSabioRoot = Split-Path $PSScriptRoot -Parent
$script:GranSabioPython = $PythonPath

function Get-GranSabioState {
    param([ValidateRange(1, 65535)][int]$Port)

    # Query failures must not be mistaken for a free port.
    $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop |
        Where-Object LocalPort -eq $Port)
    $state = [pscustomobject]@{
        Port = $Port; Status = 'free'; Process = $null; Started = $null
        Url = $null; Healthy = $false; Detail = 'Port is free.'
    }
    if ($listeners.Count -eq 0) { return $state }

    $state.Status = 'unknown'
    $state.Detail = 'Port is occupied; service not identified. Stop is disabled.'
    $owners = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -ne 1) { return $state }

    try {
        $owner = Get-Process -Id $owners[0] -ErrorAction Stop
        if ($owner.ProcessName -notin @('python', 'pythonw')) { return $state }
        $started = $owner.StartTime.ToUniversalTime().Ticks
        $address = $listeners[0].LocalAddress
        if ($address -eq '0.0.0.0') { $address = '127.0.0.1' }
        if ($address -eq '::') { $address = '::1' }
        $url = [System.UriBuilder]::new('http', $address, $Port).Uri.AbsoluteUri.TrimEnd('/')
        $info = Invoke-RestMethod -Uri "$url/api" -TimeoutSec 2 -MaximumRedirection 0
        if ($info.service -ne 'Gran Sabio LLM Engine') { return $state }
        $state.Status = 'running'
        $state.Process = $owner
        $state.Started = $started
        $state.Url = $url
        $state.Detail = "GranSabio is running (PID $($owner.Id))."
    } catch {
        return $state
    }

    try {
        $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2 -MaximumRedirection 0
        $state.Healthy = $health.status -eq 'healthy'
    } catch {
        $state.Healthy = $false
    }
    if (-not $state.Healthy) { $state.Detail += ' Health check failed.' }
    return $state
}

function Start-GranSabio {
    param([ValidateRange(1, 65535)][int]$Port)

    $state = Get-GranSabioState -Port $Port
    if ($state.Status -ne 'free') { throw "Cannot start on port ${Port}: $($state.Detail)" }
    if (-not (Test-Path -LiteralPath $script:GranSabioPython)) { throw 'Dedicated Python venv not found.' }

    $previousPort = $env:APP_PORT
    $previousReload = $env:APP_RELOAD
    try {
        $env:APP_PORT = [string]$Port
        $env:APP_RELOAD = 'false'
        $arguments = '-u "{0}" --file-logging --force-extra-verbose' -f (Join-Path $script:GranSabioRoot 'main.py')
        $child = Start-Process -FilePath $script:GranSabioPython -ArgumentList $arguments `
            -WorkingDirectory $script:GranSabioRoot -PassThru
    } finally {
        $env:APP_PORT = $previousPort
        $env:APP_RELOAD = $previousReload
    }

    Write-Host "Starting GranSabio on port $Port in a separate console..."
    $deadline = (Get-Date).AddSeconds(90)
    do {
        if ($child.HasExited) { throw "Server exited during startup. Check the logs in $script:GranSabioRoot\logs." }
        $state = Get-GranSabioState -Port $Port
        if ($state.Status -eq 'running' -and $state.Healthy) { return $state }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    throw 'Startup was not confirmed within 90 seconds. Check the server console; the process has not been stopped.'
}

function Stop-GranSabio {
    param([Parameter(Mandatory = $true)]$ExpectedState)

    # Recheck identity immediately before stopping; never kill by port alone.
    $current = Get-GranSabioState -Port $ExpectedState.Port
    if ($ExpectedState.Status -ne 'running' -or $current.Status -ne 'running' -or
        $current.Process.Id -ne $ExpectedState.Process.Id -or $current.Started -ne $ExpectedState.Started) {
        throw 'Server identity changed or could not be verified. Nothing was stopped; refresh the menu.'
    }
    Stop-Process -InputObject $current.Process -ErrorAction Stop
    if (-not $current.Process.WaitForExit(5000)) { throw 'The process has not exited yet.' }
    $after = Get-GranSabioState -Port $ExpectedState.Port
    if ($after.Status -ne 'free') { throw 'The selected process stopped, but the port is still occupied. Refresh the menu.' }
    Write-Host "GranSabio on port $($ExpectedState.Port) stopped."
}

function Read-GranSabioPort {
    $answer = Read-Host 'Port (1-65535)'
    $selected = 0
    if (-not [int]::TryParse($answer, [ref]$selected) -or $selected -lt 1 -or $selected -gt 65535) {
        throw 'Enter a port number between 1 and 65535.'
    }
    return $selected
}

function Show-GranSabioMenu {
    param([int]$Port)

    if ($Port -eq 0) {
        Push-Location $script:GranSabioRoot
        try {
            # Use the existing dotenv parser without importing the engine or exposing secrets.
            $configured = & $script:GranSabioPython -c "import os; from dotenv import dotenv_values; print(os.environ.get('APP_PORT', dotenv_values('.env').get('APP_PORT') or '8000'))"
            if ($LASTEXITCODE -ne 0) { throw 'Could not read the configured port.' }
        } finally {
            Pop-Location
        }
        if (-not [int]::TryParse([string]$configured, [ref]$Port) -or $Port -lt 1 -or $Port -gt 65535) {
            throw 'APP_PORT must be a number between 1 and 65535.'
        }
    }

    while ($true) {
        $state = Get-GranSabioState -Port $Port
        Write-Host "`nGranSabio - port $Port"
        Write-Host $state.Detail
        if ($state.Url) { Write-Host $state.Url }
        if ($state.Status -eq 'free') { Write-Host '1. Start this instance' }
        if ($state.Status -eq 'running') { Write-Host '1. Stop this instance (interrupts running analyses)' }
        Write-Host '2. Start an instance on another port'
        Write-Host '3. Select another port'
        Write-Host '4. Refresh status'
        Write-Host '0. Exit menu (leave servers running)'
        $choice = Read-Host 'Choose'
        try {
            switch ($choice) {
                '1' {
                    if ($state.Status -eq 'free') { $null = Start-GranSabio -Port $Port }
                    elseif ($state.Status -eq 'running') { Stop-GranSabio -ExpectedState $state }
                    else { throw $state.Detail }
                }
                '2' {
                    $newPort = Read-GranSabioPort
                    $null = Start-GranSabio -Port $newPort
                    $Port = $newPort
                }
                '3' { $Port = Read-GranSabioPort }
                '4' { }
                '0' { return }
                default { Write-Host 'Choose one of the listed options.' }
            }
        } catch {
            Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# Dot-sourcing also makes status/start/stop available to local PowerShell scripts.
if ($MyInvocation.InvocationName -ne '.') {
    try { Show-GranSabioMenu -Port $Port }
    catch {
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}
