# xmake 一键安装脚本（PowerShell）

Write-Host "====================================" -ForegroundColor Green
Write-Host "Installing xmake" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

# 检查是否已安装
$xmakeInstalled = Get-Command xmake -ErrorAction SilentlyContinue

if ($xmakeInstalled) {
    Write-Host "✓ xmake is already installed!" -ForegroundColor Green
    Write-Host "Version: " -NoNewline
    xmake --version
    Write-Host ""
    Write-Host "Run build_xmake.bat to build the project."
    exit 0
}

Write-Host "Downloading and installing xmake..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
Write-Host ""

try {
    # 下载并执行安装脚本
    Invoke-Expression (Invoke-Webrequest 'https://xmake.io/psget.txt' -UseBasicParsing).Content
    
    Write-Host ""
    Write-Host "====================================" -ForegroundColor Green
    Write-Host "✓ Installation completed!" -ForegroundColor Green
    Write-Host "====================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Please restart your terminal and run:" -ForegroundColor Cyan
    Write-Host "  build_xmake.bat" -ForegroundColor Cyan
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "====================================" -ForegroundColor Red
    Write-Host "✗ Installation failed!" -ForegroundColor Red
    Write-Host "====================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative installation methods:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Scoop (recommended):" -ForegroundColor Cyan
    Write-Host "   scoop install xmake" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Manual download:" -ForegroundColor Cyan
    Write-Host "   https://github.com/xmake-io/xmake/releases" -ForegroundColor White
    Write-Host ""
    exit 1
}

Read-Host "Press Enter to exit"

