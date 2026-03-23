param(
  [string]$ServerIp = "65.109.128.235",
  [string]$RemoteDir = "/root/tetris-rl-ppo",
  [int]$Episodes = 5,
  [double]$Sleep = 0.08,
  [bool]$Hold = $true,
  [string]$Prefix = "dqn_tetris_macro",
  [ValidateSet("features","board")][string]$Obs = "features",
  [bool]$HoldActions = $false,
  [int]$NextN = 1,
  [ValidateSet("tetris","survival","balanced")][string]$RewardProfile = "tetris"
)

$ErrorActionPreference = "Stop"

function Get-PythonLauncher {
  $py = Get-Command python -ErrorAction SilentlyContinue
  if ($py) {
    return @{ Exe = $py.Source; Args = @() }
  }

  $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
  if ($pyLauncher) {
    return @{ Exe = $pyLauncher.Source; Args = @("-3") }
  }

  throw "Could not find 'python' or 'py' on PATH. Install Python 3 and try again."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$localModelsDir = Join-Path $repoRoot "models"
New-Item -ItemType Directory -Force -Path $localModelsDir | Out-Null

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
  throw "ssh not found on PATH. Install OpenSSH client and try again."
}
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
  throw "scp not found on PATH. Install OpenSSH client and try again."
}

Write-Host "==> Finding latest DQN checkpoint(s) on server ${ServerIp}..."
$remoteListCmd = "ls -1t ${RemoteDir}/models/${Prefix}*_steps.zip 2>/dev/null | head -n 5"
$remotePaths = @(ssh -o ConnectTimeout=10 "root@${ServerIp}" $remoteListCmd) | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

if ($remotePaths.Count -eq 0) {
  throw "No DQN .zip checkpoints found on server at ${RemoteDir}/models/ (prefix ${Prefix})."
}

$python = Get-PythonLauncher

function Download-Checkpoint([string]$RemotePath) {
  $fileName = [System.IO.Path]::GetFileName($RemotePath)
  $destPath = Join-Path $localModelsDir $fileName

  Write-Host "==> Pulling via scp: ${RemotePath}"
  & scp -o ConnectTimeout=10 "root@${ServerIp}:${RemotePath}" "${localModelsDir}" | Out-Host
  if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
  }

  return $destPath
}

function Test-ModelZip([string]$ModelPath) {
  & $python.Exe @($python.Args + @(
    "-c",
    "from stable_baselines3 import DQN; DQN.load(r'${ModelPath}'); print('ok')"
  ))

  return $LASTEXITCODE -eq 0
}

$chosenLocalPath = $null
foreach ($rp in $remotePaths) {
  try {
    $lp = Download-Checkpoint -RemotePath $rp
    Write-Host "==> Downloaded: ${lp}"

    if (Test-ModelZip -ModelPath $lp) {
      $chosenLocalPath = $lp
      break
    }

    Write-Host "WARNING: Downloaded checkpoint could not be loaded (maybe still being written). Trying previous..." -ForegroundColor Yellow
  }
  catch {
    Write-Host "WARNING: Failed to download/load ${rp}: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

if (-not $chosenLocalPath) {
  throw "Could not find a loadable checkpoint among the latest ${($remotePaths.Count)} files. Try again in ~30s."
}

Write-Host "==> Launching enjoy_local_dqn.py with: ${chosenLocalPath}"
Push-Location $scriptDir
try {
  $argsList = @(
    "enjoy_local_dqn.py",
    "--model-path", $chosenLocalPath,
    "--episodes", "${Episodes}",
    "--sleep", "${Sleep}",
    "--obs", "${Obs}",
    "--next-n", "${NextN}",
    "--reward-profile", "${RewardProfile}"
  )
  if ($HoldActions) {
    $argsList += "--hold-actions"
  }
  if ($Hold) {
    $argsList += "--hold"
  }

  & $python.Exe @($python.Args + $argsList)
}
finally {
  Pop-Location
}
