@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "LLVM_ROOT=C:\Program Files\LLVM"
if not exist "%LLVM_ROOT%\bin\clang-cl.exe" set "LLVM_ROOT=%LOCALAPPDATA%\llvm-portable\clang+llvm-23.1.2-x86_64-pc-windows-msvc"
set "CV_ROOT=C:\opencv"
if not exist "%CV_ROOT%\build\include\opencv2\core.hpp" set "CV_ROOT=%LOCALAPPDATA%\opencv-portable\opencv"
if not exist "x64\Cpu\obj" exit /b 1
"%LLVM_ROOT%\bin\clang-cl.exe" /c /O2 /EHsc /MD /std:c++17 /I "%CV_ROOT%\build\include" /I ImgReconstruct_backend /I third_party\onnxruntime\include tests\test_seg_blend.cpp /Fo:x64\Cpu\obj\test_seg_blend.obj
if errorlevel 1 exit /b 1
link /OUT:x64\Cpu\test_seg_blend.exe /DELAYLOAD:onnxruntime.dll x64\Cpu\obj\test_seg_blend.obj x64\Cpu\obj\CS_encryption.obj x64\Cpu\obj\helper_functions.obj x64\Cpu\obj\crypto_utils.obj x64\Cpu\obj\cs_wavelet.obj x64\Cpu\obj\cs_dict.obj x64\Cpu\obj\cs_gpu.obj x64\Cpu\obj\image_tiles.obj /LIBPATH:"%CV_ROOT%\build\x64\vc16\lib" /LIBPATH:"%LLVM_ROOT%\lib" /LIBPATH:third_party\onnxruntime\lib opencv_world490.lib bcrypt.lib libomp.lib onnxruntime.lib delayimp.lib user32.lib shell32.lib ole32.lib
if errorlevel 1 exit /b 1
x64\Cpu\test_seg_blend.exe
