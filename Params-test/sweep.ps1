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
    [switch]$PerTileCoef,
    [switch]$Metrics
)

$ErrorActionPreference = "Continue"
$csv = Join-Path $OutDir "results.csv"
if (!(Test-Path $csv)) {
    "geometry,ratio,sampling,tiles,coef,tv,pertile,psnr,seconds,enc_bytes" | Out-File $csv -Encoding utf8
}
$ptflag = "0"
if ($PerTileCoef) { $ptflag = "1" }
$done = @{}
$allrows = @(Import-Csv $csv)
foreach ($rw in $allrows) {
    $k = "$($rw.geometry)|$($rw.ratio)|$($rw.sampling)|$($rw.tiles)|$($rw.coef)|$($rw.tv)|$($rw.pertile)"
    if ($Metrics) {
        if ($rw.ssim -ne "" -and $rw.ssim -ne $null) { $done[$k] = $true }
    } else {
        $done[$k] = $true
    }
}
$gridkeys = @{}
foreach ($g in $Geometries) { foreach ($r in $Ratios) { foreach ($s in $Samplings) {
    foreach ($t in $Tiles) { foreach ($c in $Coefs) { foreach ($v in $Tvs) {
        $gridkeys["$g|$r|$s|$t|$c|$v|$ptflag"] = $true
    }}}
}}}
if ($Metrics) {
    # drop metric-less rows INSIDE the current grid only; they get
    # recomputed below (decrypt + all metrics). Rows outside the grid
    # (other ratios/tiles) are preserved untouched.
    $kept = @($allrows | Where-Object {
        $k = "$($_.geometry)|$($_.ratio)|$($_.sampling)|$($_.tiles)|$($_.coef)|$($_.tv)|$($_.pertile)"
        (-not $gridkeys.ContainsKey($k)) -or ($_.ssim -ne "" -and $_.ssim -ne $null)
    })
    $hdr = "geometry,ratio,sampling,tiles,coef,tv,pertile,psnr,seconds,enc_bytes,ssim,de_mean,de_p95,lpips"
    @($hdr) + @($kept | ForEach-Object {
        "$($_.geometry),$($_.ratio),$($_.sampling),$($_.tiles),$($_.coef),$($_.tv),$($_.pertile),$($_.psnr),$($_.seconds),$($_.enc_bytes),$($_.ssim),$($_.de_mean),$($_.de_p95),$($_.lpips)"
    }) | Out-File $csv -Encoding utf8
}

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
                        $ssim = ""
                        $demean = ""
                        $dep95 = ""
                        $lpips = ""
                        if (Test-Path $dec) {
                            $o = & $Exe compare $Image $dec $cmp --amp 4 2>$null | Out-String
                            $m = [regex]::Match($o, 'PSNR:\s+([0-9.]+) dB')
                            if ($m.Success) { $psnr = $m.Groups[1].Value }
                            if ($Metrics) {
                                $py = 'C:\Users\iorda\AppData\Local\Programs\Python\Python312\python.exe'
                                $mo = & $py -u (Join-Path $OutDir "metrics.py") $Image $dec 2>$null | Out-String
                                $ms = [regex]::Match($mo, 'SSIM:\s+([0-9.]+)')
                                if ($ms.Success) { $ssim = $ms.Groups[1].Value }
                                $md = [regex]::Match($mo, 'CIEDE2000 mean:\s+([0-9.]+)')
                                if ($md.Success) { $demean = $md.Groups[1].Value }
                                $mp = [regex]::Match($mo, 'p95:\s+([0-9.]+)')
                                if ($mp.Success) { $dep95 = $mp.Groups[1].Value }
                                $lo = & $py -u (Join-Path $OutDir "metrics_lpips.py") $Image $dec 2>$null | Out-String
                                $ml = [regex]::Match($lo, 'LPIPS-\w+:\s+([0-9.]+)')
                                if ($ml.Success) { $lpips = $ml.Groups[1].Value }
                            }
                            Remove-Item $dec -ErrorAction SilentlyContinue
                        }
                        "$g,$r,$s,$t,$c,$v,$ptflag,$psnr,$([math]::Round($dt,1)),$encBytes,$ssim,$demean,$dep95,$lpips" | Out-File $csv -Encoding utf8 -Append
                    }
                }
            }
        }
    }
}
Write-Output "done. results in $csv"
