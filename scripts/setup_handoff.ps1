param([switch]$Full)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
function Run([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $Arguments" }
}
function EnvPython([string]$Relative) {
    $path = Join-Path $projectRoot $Relative
    if (-not (Test-Path -LiteralPath (Join-Path $path 'Scripts/python.exe'))) {
        Run 'py' @('-3.12', '-m', 'venv', $path)
    }
    return (Join-Path $path 'Scripts/python.exe')
}
Run 'py' @('-3.12', '-c', 'import sys; assert sys.version_info[:2] == (3, 12)')
Run 'node' @('-e', 'const [m,n]=process.versions.node.split(String.fromCharCode(46)).map(Number); if(!((m===20&&n>=19)||(m===22&&n>=12)||m>22))process.exit(1)')
$tv1Python = EnvPython 'tv1/.venv'
Run $tv1Python @('-m', 'pip', 'install', '-r', 'tv1/requirements.txt')
$tv3Python = EnvPython 'tv3/WP04/.venv'
Run $tv3Python @('-m', 'pip', 'install', '-e', './tv3/WP04[api]')
$tv4Python = EnvPython 'tv4/.venv'
Run $tv4Python @('-m', 'pip', 'install', '-e', './tv4[api]', '-r', 'tv4/requirements.txt', 'httpx', 'pyarrow', '-e', './tv2/WP08', '-e', './tv2/WP09')
$coordinatorPython = EnvPython 'tv2/WP03/.venvs/coordinator'
Run $coordinatorPython @('-m', 'pip', 'install', '-e', './tv2/WP03')
$modelNames = @('metaclip2')
if ($Full) { $modelNames += @('beit3', 'bge_vl', 'perception') }
foreach ($modelName in $modelNames) {
    $workerPython = EnvPython "tv2/WP03/.venvs/$modelName"
    Run $workerPython @('-m', 'pip', 'install', 'torch==2.11.0', 'torchvision==0.26.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
    Run $workerPython @('-m', 'pip', 'install', '-r', "tv2/WP03/envs/$modelName.txt", '-e', './tv2/WP03')
    if ($modelName -eq 'perception') {
        # Upstream training dependencies include packages without Windows wheels.
        Run $workerPython @('-m', 'pip', 'install', '--no-deps', 'git+https://github.com/facebookresearch/perception_models.git@3e352cca660658d4b5c90f42a7808b11469e4c66')
        Run $workerPython @('-c', 'import core.vision_encoder.pe; import core.vision_encoder.transforms')
    }
    Run $workerPython @('-c', 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))')
}
Push-Location tv5
try { Run 'npm.cmd' @('ci'); Run 'npm.cmd' @('run', 'build') } finally { Pop-Location }
Write-Host 'Environment setup complete. Continue docs/RUN_FROM_DATASETS_ZIP.md (models, data, index).'
