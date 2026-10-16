<#PSScriptInfo>
.VERSION 1.0
.GUID 9f3a1c2e-7b4d-4e8a-9c1f-2d5b6a7c8e9f
.AUTHOR params
.DESCRIPTION Grid-search coef x tv x tiles x ratio x geometry for seg-auto sampling, score by PSNR.
#>
<#
.SYNOPSIS
  Grid-search decrypt coef/tv across ratios, tile grids and geometries.
.DESCRIPTION
  For every (geometry, ratio, sampling, tiles, coef, tv) cell this encrypts
  once per (geometry, ratio, sampling) and decrypts per cell, scoring with
  the built-in compare. Results append to results.csv (resumable: completed
  cells are skipped). Timings recorded per cell.
.EXAMPLE
  .\Params-test\sweep.ps1 -Password "testtesttest12"
.EXAMPLE
  .\Params-test\sweep.ps1 -Geometries @("half") -Ratios @(0.3) -Tiles @(64) -Coefs @(0.035) -Tvs @(0.01)
#>
[CmdletBinding()]
param(
    [string]$Exe = "x64\Cpu\ImgReconstruct_backend.exe",
    [string]$Image = "ImgReconstruct_backend\IMG_3690.png",
    [string]$Password = "param-sweep-1234",
    [string]$OutDir = "Params-test",
    [string[]]$Geometries = @("half", "full"),
    [double[]]$Ratios = @(0.15, 0.3, 0.5),
    [string[]]$Samplings = @("seg", "uniform"),
    [int[]]$Tiles = @(64, 96, 128),
    [double[]]$Coefs = @(0.01, 0.02, 0.035, 0.05, 0.08),
    [double[]]$Tvs = @(0.0, 0.01, 0.03),
    [int]$Iterations = 8,
    [int]$Overlap = 24,
    [switch]$PerTileCoef
)

$ErrorActionPreference = "Continue"
$csv = Join-Path $OutDir "results.csv"
if (!(Test-Path $csv)) {
    "geometry,ratio,sampling,tiles,coef,tv,pertile,psnr,seconds,enc_bytes" | Out-File $csv -Encoding utf8
}
$done = @{}
Import-Csv $csv | ForEach-Object { $done["$($_.geometry)|$($_.ratio)|$($_.sampling)|$($_.tiles)|$($_.coef)|$($_.tv)|$($_.pertile)"] = $true }
$ptflag = "0"
if ($PerTileCoef) { $ptflag = "1" }

$ncells = 0
foreach ($g in $Geometries) { foreach ($r in $Ratios) { foreach ($s in $Samplings) {
    $ncells += $Tiles.Count * $Coefs.Count * $Tvs.Count
}}}
Write-Output "grid cells: $ncells (skipping $($done.Count) done)"
$cell = 0
foreach ($g in $Geometries) {
    $fr = @()
    if ($g -eq "full") { $fr = @("--full-res") }
    $tag = $g
    foreach ($r in $Ratios) {
        foreach ($s in $Samplings) {
            $stag = "${tag}-r${r}-${s}"
            $enc = Join-Path $OutDir "enc-$stag.enc.png"
            if (!(Test-Path $enc)) {
                $sargs = @("encrypt", $Image, $enc, "--password", $Password,
                    "--ratio", "$r") + $fr + @("--no-preview")
                if ($s -eq "seg") { $sargs += @("--seg-auto", "8") }
                & $Exe @sargs 2>$null
            }
            $encBytes = (Get-Item $enc).Length
            foreach ($t in $Tiles) {
                foreach ($c in $Coefs) {
                    foreach ($v in $Tvs) {
                        $cell++
                        $key = "$g|$r|$s|$t|$c|$v|$ptflag"
                        if ($done.ContainsKey($key)) { continue }
                        Write-Output "[$cell/$ncells] $key"
                        $dec = Join-Path $OutDir "dec-tmp.dec.png"
                        $cmp = Join-Path $OutDir "cmp-tmp"
                        $t0 = Get-Date
                        $dargs = @("decrypt", $enc, $dec, "--password", $Password,
                            "--manual", "--tiles", "$t", "--overlap", "$Overlap",
                            "--iterations", "$Iterations", "--coef", "$c",
                            "--tv", "$v", "--no-preview")
                        if ($g -eq "full") { $dargs += @("--full-res") }
                        if ($PerTileCoef) { $dargs += @("--per-tile-coef") }
                        & $Exe @dargs 2>$null
                        $dt = ((Get-Date) - $t0).TotalSeconds
                        $psnr = ""
                        if (Test-Path $dec) {
                            $o = & $Exe compare $Image $dec $cmp --amp 4 2>$null | Out-String
                            $m = [regex]::Match($o, 'PSNR:\s+([0-9.]+) dB')
                            if ($m.Success) { $psnr = $m.Groups[1].Value }
                            Remove-Item $dec -ErrorAction SilentlyContinue
                        }
                        "$g,$r,$s,$t,$c,$v,$ptflag,$psnr,$([math]::Round($dt,1)),$encBytes" | Out-File $csv -Encoding utf8 -Append
                    }
                }
            }
        }
    }
}
Write-Output "done. results in $csv"
