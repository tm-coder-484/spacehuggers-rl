# temp_monitor.ps1 — live CPU temp + clock-throttle monitor.  NO ADMIN REQUIRED.
#
# Launched by train.bat / train_game.py when you pass --v.  Runs in its own
# window so it doesn't flood the training output (which prints only every ~40s).
#
# What it shows:
#   zone    — ACPI thermal-zone temperature.  NOTE: this is NOT the true CPU
#             core/package temp (that needs an admin driver).  Real core temp
#             runs ~20-35 C hotter than this zone.  Use it for trend, not absolute.
#   clock   — effective CPU frequency = base x (% Processor Performance).
#             THIS is the real throttle signal: below base clock = throttling.
#   vs base — % of base clock.  >100 = turbo, <100 = throttling.

$host.UI.RawUI.WindowTitle = 'train :: temp + clock monitor'
$base = (Get-CimInstance Win32_Processor).MaxClockSpeed

Write-Host ''
Write-Host '  Live CPU monitor  (no-admin: ACPI zone + perf counters)' -ForegroundColor Cyan
Write-Host '  zone temp is the ACPI sensor; real core temp runs ~20-35C hotter.' -ForegroundColor DarkGray
Write-Host ("  Base clock {0} MHz.  Clock below base = throttling.  Ctrl+C to close.`n" -f $base) -ForegroundColor DarkGray
Write-Host ('  {0,-8} {1,7} {2,10} {3,8}  {4}' -f 'time','zone','clock','vs base','status') -ForegroundColor Gray

$counters = @(
    '\Thermal Zone Information(*)\Temperature',
    '\Processor Information(_Total)\% Processor Performance'
)

while ($true) {
    $tempC = '--'; $mhz = '--'; $pct = 0
    try {
        $samples = (Get-Counter -Counter $counters -ErrorAction Stop).CounterSamples
        $tz = $samples | Where-Object { $_.Path -like '*thermal*' } |
              Sort-Object CookedValue -Descending | Select-Object -First 1
        if ($tz) { $tempC = [math]::Round($tz.CookedValue - 273.15, 1) }
        $pf = $samples | Where-Object { $_.Path -like '*processor performance*' } | Select-Object -First 1
        if ($pf) { $pct = [math]::Round($pf.CookedValue); $mhz = [math]::Round($base * $pf.CookedValue / 100) }
    } catch { }

    if     ($pct -ge 98) { $status = 'OK';            $col = 'Green'  }
    elseif ($pct -ge 85) { $status = 'mild throttle'; $col = 'Yellow' }
    else                 { $status = 'THROTTLING';    $col = 'Red'    }

    $line = '  {0,-8} {1,5} C {2,7} MHz {3,6}%  {4}' -f `
        (Get-Date -Format 'HH:mm:ss'), $tempC, $mhz, $pct, $status
    Write-Host $line -ForegroundColor $col
    Start-Sleep -Seconds 2
}
