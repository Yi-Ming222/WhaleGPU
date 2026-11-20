param([ValidateSet('smoke','vector','shared','shuffle','all')][string]$Stage = 'all')
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + $Stage + '-' + [guid]::NewGuid().ToString('N').Substring(0,8)
$runDir = Join-Path $projectRoot "results\$runId"
$buildDir = Join-Path $projectRoot "build\$runId"
New-Item -ItemType Directory -Path $runDir,$buildDir | Out-Null
$exitCode = 2
$status = 'error'
$started = [DateTimeOffset]::Now.ToString('o')
try {
    $snapshot = Join-Path $runDir 'source'
    New-Item -ItemType Directory -Path $snapshot | Out-Null
    foreach ($item in @('include','src','tests','build.cmd','run-tests.ps1')) {
        Copy-Item -LiteralPath (Join-Path $projectRoot $item) -Destination $snapshot -Recurse
    }
    Get-ChildItem -LiteralPath $snapshot -Recurse -File | Get-FileHash -Algorithm SHA256 |
        Select-Object Path,Hash | Export-Csv (Join-Path $runDir 'source-sha256.csv') -NoTypeInformation
    (& nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total --format=csv 2>&1) |
        Set-Content (Join-Path $runDir 'nvidia-smi.csv') -Encoding UTF8
    & (Join-Path $projectRoot 'build.cmd') $buildDir 2>&1 | Tee-Object -FilePath (Join-Path $runDir 'build.log')
    if ($LASTEXITCODE -ne 0) { throw "nvcc build failed: $LASTEXITCODE" }
    $binary = Join-Path $buildDir 'whalegpu_tests.exe'
    Get-FileHash -LiteralPath $binary -Algorithm SHA256 | Export-Csv (Join-Path $runDir 'binary-sha256.csv') -NoTypeInformation
    & $binary $Stage $runDir 2>&1 | Tee-Object -FilePath (Join-Path $runDir 'test.log')
    $exitCode = $LASTEXITCODE
    $status = if ($exitCode -eq 0) { 'PASS' } else { 'FAIL' }
} catch {
    $_ | Out-String | Set-Content (Join-Path $runDir 'error.log') -Encoding UTF8
    Write-Host $_
} finally {
    [ordered]@{
        run_id=$runId; stage=$Stage; started_at=$started; finished_at=[DateTimeOffset]::Now.ToString('o')
        status=$status; exit_code=$exitCode; build_directory=$buildDir
        os=[Environment]::OSVersion.VersionString; cpu=$env:PROCESSOR_IDENTIFIER
        timing='CUDA Events; 2 warmups; 5 samples x 10 repeats; full reduction pass chain; excludes allocation/copies/CPU reference'
    } | ConvertTo-Json | Set-Content (Join-Path $runDir 'run.json') -Encoding UTF8
    Write-Host "Run artifacts: $runDir"
}
exit $exitCode
