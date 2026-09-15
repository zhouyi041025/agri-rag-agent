# 荟诊 · 一键启动脚本
# 用法：在项目根目录执行  .\run.ps1
# 首次运行会自动创建虚拟环境、安装依赖、生成示例数据并构建索引。

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Port = if ($env:AGRI_PORT) { $env:AGRI_PORT } else { "8000" }
$Url = "http://127.0.0.1:$Port"

# 1) 准备虚拟环境
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/4] 创建虚拟环境 .venv ..." -ForegroundColor Cyan
    python -m venv (Join-Path $PSScriptRoot ".venv")
}

# 2) 安装依赖（能导入核心库就跳过）
& $VenvPython -c "import fastapi, uvicorn, numpy, jieba" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[2/4] 安装依赖 ..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip -q
    & $VenvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
} else {
    Write-Host "[2/4] 依赖已就绪" -ForegroundColor DarkGray
}

# 3) 生成示例环境数据并构建索引
if (-not (Test-Path (Join-Path $PSScriptRoot "data\env\field_env.csv"))) {
    Write-Host "[3/4] 生成田间环境示例数据 ..." -ForegroundColor Cyan
    & $VenvPython (Join-Path $PSScriptRoot "scripts\make_env_data.py")
}
if (-not (Test-Path (Join-Path $PSScriptRoot "artifacts\index\chunks.jsonl"))) {
    Write-Host "[3/4] 构建向量索引 ..." -ForegroundColor Cyan
    $env:PYTHONPATH = Join-Path $PSScriptRoot "src"
    & $VenvPython -m agri_agent.cli build
} else {
    Write-Host "[3/4] 索引已就绪" -ForegroundColor DarkGray
}

# 4) 启动服务，健康检查通过后再打开浏览器
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
Write-Host "[4/4] 正在启动服务 $Url ..." -ForegroundColor Cyan
$proc = Start-Process -FilePath $VenvPython `
    -ArgumentList @("-m", "uvicorn", "agri_agent.serving.api:app", "--app-dir", "src", "--port", $Port) `
    -PassThru -NoNewWindow

try {
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $resp = Invoke-WebRequest -Uri "$Url/health" -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
    }
    if ($ready) {
        Write-Host "服务已启动：$Url   按 Ctrl+C 停止" -ForegroundColor Green
        Start-Process $Url | Out-Null
    } else {
        Write-Host "服务启动超时，请查看上方日志。" -ForegroundColor Yellow
    }
    Wait-Process -Id $proc.Id
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
