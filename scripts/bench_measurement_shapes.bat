@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "LLVM_ROOT=C:\Program Files\LLVM"
if not exist "%LLVM_ROOT%\bin\clang-cl.exe" set "LLVM_ROOT=%LOCALAPPDATA%\llvm-portable\clang+llvm-23.1.2-x86_64-pc-windows-msvc"
set "CV_ROOT=C:\opencv"
if not exist "%CV_ROOT%\build\include\opencv2\core.hpp" set "CV_ROOT=%LOCALAPPDATA%\opencv-portable\opencv"
if not exist "x64\Cpu\obj" exit /b 1
"%LLVM_ROOT%\bin\clang-cl.exe" /O2 /EHsc /MD /std:c++17 /I "%CV_ROOT%\build\include" tests\bench_measurement_shapes.cpp /Fo:x64\Cpu\obj\bench_measurement_shapes.obj /Fe:x64\Cpu\bench_measurement_shapes.exe /link /LIBPATH:"%CV_ROOT%\build\x64\vc16\lib" opencv_world490.lib
if errorlevel 1 exit /b 1
x64\Cpu\bench_measurement_shapes.exe %*
