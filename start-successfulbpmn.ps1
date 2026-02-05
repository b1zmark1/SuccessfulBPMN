param(
  [switch]$BuildLocal,
  [switch]$NoPull,
  [switch]$FollowLogs
)

$ErrorActionPreference = "Stop"

function Test-CommandExists {
  param([string]$Name)
  return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "[start] SuccessfulBPMN bootstrap..."

if (-not (Test-CommandExists "docker")) {
  throw "Docker CLI not found. Install Docker Desktop and retry."
}

try {
  docker info | Out-Null
}
catch {
  throw "Docker daemon is not running. Start Docker Desktop and retry."
}

# Resolve project root robustly whether script is in repo root or in scripts/.
if (Test-Path (Join-Path $PSScriptRoot "docker-compose.yml")) {
  $repoRoot = $PSScriptRoot
}
elseif (Test-Path (Join-Path (Split-Path -Parent $PSScriptRoot) "docker-compose.yml")) {
  $repoRoot = Split-Path -Parent $PSScriptRoot
}
else {
  throw "docker-compose.yml was not found near script path: $PSScriptRoot"
}

Set-Location $repoRoot

$composeBase = @("-f", "docker-compose.yml")
$composePublished = @("-f", "docker-compose.published.yml")

if ($BuildLocal) {
  Write-Host "[start] Mode: local build"
  if (-not $NoPull) {
    docker compose @composeBase pull postgres redis
  }
  docker compose @composeBase up -d --build
}
else {
  Write-Host "[start] Mode: published images (backend/worker from Docker Hub)"
  if (-not (Test-Path "docker-compose.published.yml")) {
    throw "docker-compose.published.yml was not found in $repoRoot"
  }
  if (-not $NoPull) {
    docker compose @composeBase @composePublished pull
  }
  docker compose @composeBase @composePublished up -d
}

Write-Host "[start] Services status:"
docker compose @composeBase ps

if ($FollowLogs) {
  Write-Host "[start] Streaming backend/worker logs (Ctrl+C to stop)..."
  docker compose @composeBase logs -f backend worker
}
else {
  Write-Host "[start] Done. Open frontend separately:"
  Write-Host "        cd frontend; npm install; npm run dev"
}
