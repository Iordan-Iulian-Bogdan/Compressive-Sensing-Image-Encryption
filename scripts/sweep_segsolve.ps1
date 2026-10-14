<#PSScriptInfo>
.VERSION 1.0
.GUID 7a3c9e21-4b5f-4a8d-9c2e-1f6a8b3d4e5c
.AUTHOR sweep
.DESCRIPTION Sweep per-segment solver params across compression ratios for seg-solve containers.
#>
<#
.SYNOPSIS
  Grid-search seg-solve solver params x compression ratios, score by PSNR.

.DESCRIPTION
  For every (ratio, bird_iters, bird_coef, bird_tv) combo this runs a
  seg-mask roundtrip with isolated per-class solves (--seg-solve) and scores
  the decrypt with the built-in compare (PSNR + edge-PSNR). With
  -IncludeJoint it also scores the standard joint solve per ratio as a
  baseline. Results append to a CSV; at the end the best PSNR row per ratio
  is printed. Reruns are idempotent (same password -> same container).

  Cost: one full seg-solve per grid cell (one solve per mask class), so
  default to a small image (the thumb) for grid search and validate the
  winner on full resolution. Timings are recorded per cell.

.EXAMPLE
  .\scripts\sweep_segsolve.ps1 -Password "testtesttest12"
.EXAMPLE
  .\scripts\sweep_segsolve.ps1 -Ratios @(0.15, 0.3) -BirdIters @(6, 8, 10) -BirdCoef @(0.03, 0.045) -IncludeJoint
#>
[CmdletBinding()]
param(
  [string]$Exe = "x64\Cpu\ImgReconstruct_backend.exe",
  [string]$Image = "ImgReconstruct_backend\IMG_3690_thumb.png",
  [string]$Mask = "scene.colors.png",
  [string]$Rates = "0.1,0.8,0.1,0.1,0.2,0.3,0.2,0.2",
  [string]$Password = $env:CS_PASSWORD,
  [double[]]$Ratios = @(0.1, 0.2, 0.35, 0.5),
  [int[]]$BirdIters = @(4, 6, 8),
  [double[]]$BirdCoef = @(0.02, 0.03, 0.045),
  [double[]]$BirdTv = @(0.0, 0.05),
  [int]$BirdClass = 1,
  [int]$BgIters = 2,
  [string]$OutCsv = "sweep_segsolve.csv",
  [string]$ArtifactPrefix = "sweep",
  [switch]$IncludeJoint,
  [switch]$KeepArtifacts
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrEmpty($Password)) { throw "Password required: -Password or `$env:CS_PASSWORD" }
if (!(Test-Path $Exe)) { throw "Binary not found: $Exe (run .\build_cpu.bat main first)" }
if (!(Test-Path $Image)) { throw "Image not found: $Image" }
if (!(Test-Path $Mask)) { throw "Mask not found: $Mask" }

function Invoke-Cs([string[]]$ArgList) {
  # Route through cmd so the exe's benign stderr warnings (ref.png) merge
  # to stdout there (2>&1 is interpreted by cmd): PowerShell 5.1 then sees
  # pure stdout and never promotes stderr to error records. $LASTEXITCODE
  # carries cmd's code, which is the exe's code.
  $cmdline = ($ArgList | ForEach-Object { '"' + ($_ -replace '"', '""') + '"' }) -join ' '
  $out = cmd /d /c "$Exe $cmdline 2>&1" | ForEach-Object { "$_" }
  return @{ Out = ($out -join "`n"); Code = $LASTEXITCODE }
}

function Get-Scores([string]$Text) {
  $psnr = $null; $edge = $null
  $m = [regex]::Match($Text, "PSNR:\s+([\d.]+)\s+dB")
  if ($m.Success) { $psnr = [double]$m.Groups[1].Value }
  $m = [regex]::Match($Text, "Edge-PSNR.*?:\s+([\d.]+)\s+dB")
  if ($m.Success) { $edge = [double]$m.Groups[1].Value }
  return @{ Psnr = $psnr; Edge = $edge }
}

$cells = @()
foreach ($r in $Ratios) {
  if ($IncludeJoint) {
    $cells += @{ Ratio = $r; Mode = "joint"; BirdIters = $null; BirdCoef = $null; BirdTv = $null }
  }
  foreach ($bi in $BirdIters) {
    foreach ($bc in $BirdCoef) {
      foreach ($bt in $BirdTv) {
        $cells += @{ Ratio = $r; Mode = "segsolve"; BirdIters = $bi; BirdCoef = $bc; BirdTv = $bt }
      }
    }
  }
}
Write-Host ("Grid: {0} cells ({1} ratios)" -f $cells.Count, $Ratios.Count)

$rows = @()
$n = 0
foreach ($c in $cells) {
  $n++
  $tag = "{0}_r{1}" -f $ArtifactPrefix, ([string]$c.Ratio).Replace(".", "p")
  if ($c.Mode -eq "segsolve") {
    $sp = "{0}:iterations={1},coef={2},tv={3}" -f $BirdClass, $c.BirdIters, $c.BirdCoef, $c.BirdTv
    if ($BirdClass -ne 0) { $sp += ";0:iterations=$BgIters" }
    $tag += "_bi{0}_bc{1}_bt{2}" -f $c.BirdIters, ([string]$c.BirdCoef).Replace(".", "p"), ([string]$c.BirdTv).Replace(".", "p")
  } else {
    $sp = $null
  }
  $dec = "$tag.dec.png"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  if ($c.Mode -eq "segsolve") {
    $r = Invoke-Cs -ArgList @("roundtrip", $Image, $dec, "--ratio", ([string]$c.Ratio),
      "--seg-mask", $Mask, "--seg-rates", $Rates,
      "--seg-solve", $Mask, "--seg-params", $sp, "--no-preview", "--password", $Password)
  } else {
    $r = Invoke-Cs -ArgList @("roundtrip", $Image, $dec, "--ratio", ([string]$c.Ratio),
      "--seg-mask", $Mask, "--seg-rates", $Rates,
      "--no-preview", "--password", $Password)
  }
  if ($r.Code -ne 0) { Write-Warning ("cell {0}/{1} roundtrip failed, skipping" -f $n, $cells.Count); continue }
  $cmp = Invoke-Cs -ArgList @("compare", $Image, $dec, $tag, "--amp", "4")
  $sw.Stop()
  $s = Get-Scores $cmp.Out
  if ($null -eq $s.Psnr) { Write-Warning ("cell {0}/{1} compare failed, skipping" -f $n, $cells.Count); continue }
  $rows += [pscustomobject]@{
    ratio = $c.Ratio; mode = $c.Mode; bird_class = $BirdClass
    bird_iters = $c.BirdIters; bird_coef = $c.BirdCoef; bird_tv = $c.BirdTv; bg_iters = $BgIters
    psnr = $s.Psnr; edge_psnr = $s.Edge; time_s = [math]::Round($sw.Elapsed.TotalSeconds, 1); dec = $dec
  }
  Write-Host ("[{0}/{1}] ratio={2} {3} bi={4} bc={5} bt={6} PSNR={7} edge={8} ({9}s)" -f
    $n, $cells.Count, $c.Ratio, $c.Mode, $c.BirdIters, $c.BirdCoef, $c.BirdTv, $s.Psnr, $s.Edge,
    [math]::Round($sw.Elapsed.TotalSeconds, 1))
  if (-not $KeepArtifacts) {
    Remove-Item $dec -ErrorAction SilentlyContinue
    Remove-Item ("{0}_error.png" -f $tag) -ErrorAction SilentlyContinue
    Remove-Item ("{0}_sidebyside.png" -f $tag) -ErrorAction SilentlyContinue
  }
}

if ($rows.Count -eq 0) { throw "No successful cells; nothing to report" }
$rows | Export-Csv -Path $OutCsv -NoTypeInformation
Write-Host ""
Write-Host ("Wrote {0} ({1} rows)" -f $OutCsv, $rows.Count)
Write-Host ""
Write-Host "Best PSNR per ratio:"
$rows | Group-Object ratio | ForEach-Object {
  $_.Group | Sort-Object psnr -Descending | Select-Object -First 1
} | Sort-Object ratio | Format-Table ratio, mode, bird_iters, bird_coef, bird_tv, psnr, edge_psnr, time_s -AutoSize
