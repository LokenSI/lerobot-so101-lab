param([Parameter(Mandatory=$true)][string]$OutputPath, [Parameter(Mandatory=$true)][string]$EvaluationPath, [int]$Seconds=900)
$ErrorActionPreference='Stop'
$officeSamples=[System.Collections.Generic.List[object]]::new()
$officeDeadline=[DateTime]::UtcNow.AddSeconds($Seconds)
while ([DateTime]::UtcNow -lt $officeDeadline) {
    $officeGpu = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | Select-Object -First 1) -split ','
    $officeOs = Get-CimInstance Win32_OperatingSystem
    $officeRow=[pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o');gpu_used_mib=[int]$officeGpu[0];gpu_total_mib=[int]$officeGpu[1];host_used_mib=[Math]::Round(($officeOs.TotalVisibleMemorySize-$officeOs.FreePhysicalMemory)/1024,1);host_total_mib=[Math]::Round($officeOs.TotalVisibleMemorySize/1024,1)}
    $officeSamples.Add($officeRow)
    $officeReport=[pscustomobject]@{scope='Windows whole-device GPU and whole-system physical RAM during local checkpoint evaluation';gpu_peak_mib=($officeSamples|Measure-Object gpu_used_mib -Maximum).Maximum;host_peak_mib=($officeSamples|Measure-Object host_used_mib -Maximum).Maximum;samples=$officeSamples;threshold_14gib_exceeded=($officeSamples|Where-Object gpu_used_mib -ge 14336).Count -gt 0;threshold_48gib_ram_exceeded=($officeSamples|Where-Object host_used_mib -ge 49152).Count -gt 0}
    # Readers may hold the destination briefly through the WSL filesystem bridge.
    # Never truncate the JSON they are reading; retry an atomic replacement.
    $officeTemporary=$OutputPath+'.writing'
    [IO.File]::WriteAllText($officeTemporary,($officeReport|ConvertTo-Json -Depth 5))
    $officeWritten=$false
    for ($officeRetry=0; $officeRetry -lt 20; $officeRetry++) {
        try {
            if ([IO.File]::Exists($OutputPath)) {[IO.File]::Replace($officeTemporary,$OutputPath,$OutputPath+'.previous')}
            else {[IO.File]::Move($officeTemporary,$OutputPath)}
            $officeWritten=$true; break
        }
        catch {Start-Sleep -Milliseconds 50}
    }
    if (-not $officeWritten) {throw 'Could not publish a fresh whole-host memory sample; admission monitoring failed'}
    if ($officeRow.gpu_used_mib -ge 14336) { Write-Output 'GPU admission threshold exceeded; stop the owned evaluation'; break }
    if ($officeRow.host_used_mib -ge 49152) {Write-Output 'RAM admission threshold exceeded; stop the owned evaluation';break}
    $officeDone=$false
    foreach ($officeName in @('report.json','load-status.json','guard-report.json')) {
        $officeStatusPath=Join-Path $EvaluationPath $officeName
        if (Test-Path -LiteralPath $officeStatusPath) { try {$officeStatus=Get-Content -LiteralPath $officeStatusPath -Raw|ConvertFrom-Json; if ($officeStatus.complete -or $officeStatus.error) {$officeDone=$true}} catch {} }
    }
    if ($officeDone) {break}
    Start-Sleep -Seconds 1
}
