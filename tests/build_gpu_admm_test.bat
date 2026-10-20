@echo off
REM Builds and runs tests\gpu_admm_test.exe (HIP gpu_admm + OpenCV 4.90 reference). Outputs to x64\Gpu.
REM The HIP object is compiled first from a clean environment with the MSVC 14.29
REM toolset (ROCm 7.2 clang breaks on newer <cmath>); the test is built with clang-cl.
setlocal
set "ROOT=%~dp0..\"
mkdir "%ROOT%x64\Gpu" 2>nul
call :hipobj
if errorlevel 1 ( echo ERROR: hipcc failed & exit /b 1 )
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul
set "LLVM_ROOT=C:\Program Files\LLVM"
if not exist "%LLVM_ROOT%\bin\clang-cl.exe" set "LLVM_ROOT=%LOCALAPPDATA%\llvm-portable\clang+llvm-23.1.2-x86_64-pc-windows-msvc"
set "PATH=%LLVM_ROOT%\bin;%PATH%"
set "CV=C:\opencv"
clang-cl /c /O2 /std:c++17 /EHsc /MD /D_CRT_SECURE_NO_WARNINGS /I %CV%\build\include /I "%ROOT%ImgReconstruct_backend" "%ROOT%tests\gpu_admm_test.cpp" /Fo"%ROOT%x64\Gpu\gpu_admm_test.obj"
if errorlevel 1 ( echo ERROR: test compile failed & exit /b 1 )
link /OUT:"%ROOT%x64\Gpu\gpu_admm_test.exe" /SUBSYSTEM:CONSOLE /LIBPATH:%CV%\build\x64\vc16\lib "/LIBPATH:C:\Program Files\AMD\ROCm\7.2\lib" "%ROOT%x64\Gpu\gpu_admm_test.obj" "%ROOT%x64\Gpu\gpu_admm.obj" opencv_world490.lib amdhip64.lib
if errorlevel 1 ( echo ERROR: link failed & exit /b 1 )
copy "%CV%\build\x64\vc16\bin\opencv_world490.dll" "%ROOT%x64\Gpu\" >nul
copy "C:\Program Files\AMD\ROCm\7.2\bin\amdhip64_7.dll" "%ROOT%x64\Gpu\" >nul
if "%~1"=="run" "%ROOT%x64\Gpu\gpu_admm_test.exe"
goto :eof

:hipobj
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29 >nul
call "C:\Program Files\AMD\ROCm\7.2\bin\hipcc.bat" --offload-arch=gfx1100 -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS -fms-runtime-lib=dll -I "%ROOT%ImgReconstruct_backend" -c "%ROOT%ImgReconstruct_backend\gpu_admm.hip" -o "%ROOT%x64\Gpu\gpu_admm.obj"
endlocal & exit /b %errorlevel%
