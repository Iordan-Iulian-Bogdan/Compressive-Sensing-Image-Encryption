@echo off
REM Builds and runs tests\gpu_tiles_test.exe (HIP, gfx1100). Outputs to x64\Gpu.
REM ROCm 7.2's clang needs an older MSVC standard library than the newest
REM toolset on this machine (14.52 fails on <cmath>), hence -vcvars_ver=14.29.
setlocal
set "ROOT=%~dp0..\"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" -vcvars_ver=14.29 >nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )
mkdir "%ROOT%x64\Gpu" 2>nul
cd /d "%ROOT%x64\Gpu"
call "C:\Program Files\AMD\ROCm\7.2\bin\hipcc.bat" --offload-arch=gfx1100 -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS -I "%ROOT%ImgReconstruct_backend" "%ROOT%ImgReconstruct_backend\gpu_tiles.hip" "%ROOT%tests\gpu_tiles_test.cpp" -o gpu_tiles_test.exe
if errorlevel 1 ( echo ERROR: hipcc failed & exit /b 1 )
if "%~1"=="run" gpu_tiles_test.exe
