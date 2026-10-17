# Run the dataset CLI using the project environment.
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = 'python'
}
Push-Location $PSScriptRoot
try {
    & $taskPython -m foodquality8 @args
    if ($LASTEXITCODE -ne 0) { throw "FoodQuality-8 failed (exit $LASTEXITCODE)" }
}
finally {
    Pop-Location
}
