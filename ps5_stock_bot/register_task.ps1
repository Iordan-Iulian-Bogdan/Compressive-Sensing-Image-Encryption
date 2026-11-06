# Run as Administrator: registers PS5 stock check every 5 minutes.
# Runs in your login session so popups + browser windows reach you.
$botDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $botDir "ps5_stock_bot.py"
$python = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
if (-not $python) { $python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source }
if (-not $python) { Write-Error "Python not found in PATH."; exit 1 }

$action = New-ScheduledTaskAction -Execute $python -Argument "`"$script`"" -WorkingDirectory $botDir
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration ([TimeSpan]::MaxValue)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
Register-ScheduledTask -TaskName "PS5 Stock Bot" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Check altex.ro PS5 stock every 5 min, popup + open page on restock." -Force
Write-Host "Registered 'PS5 Stock Bot' - runs every 5 minutes when you are logged on."
