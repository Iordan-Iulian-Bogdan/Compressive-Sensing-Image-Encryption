@echo off
REM CPU-only build for ImgReconstruct_backend (no ROCm/HIP required).
REM Mirrors build_clang.bat (LLVM 23 clang-cl + MS link, same OpenCV split):
REM   main exe  -> C:\opencv 4.90 world DLL (like ImgReconstruct_backend.vcxproj)
REM   cs_tests  -> Documents\opencv 4.13 static set w/ contrib quality
REM                (like tests\cs_tests.vcxproj)
REM Instead of the hipcc cs_gpu.hip step it compiles ImgReconstruct_backend\
REM cs_gpu_cpu.cpp (pure-CPU cs_gpu interface stub: every solve runs the
REM registered CPU fallback inline), and the HIP libs/DLLs are dropped.
REM Outputs to x64\Cpu so other binaries are untouched.
REM Prereqs: LLVM 23 (winget LLVM.LLVM or portable %LOCALAPPDATA%\llvm-portable),
REM          VS2022 MSVC, OpenCV 4.90 (C:\opencv or portable
REM          %LOCALAPPDATA%\opencv-portable\opencv). No ROCm needed.
REM Usage: build_cpu.bat [main]  -- 'main' builds only ImgReconstruct_backend.exe
REM          and skips cs_tests.exe (which needs the static 4.13 contrib set).
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )
set "LLVM_ROOT=C:\Program Files\LLVM"
if not exist "%LLVM_ROOT%\bin\clang-cl.exe" set "LLVM_ROOT=%LOCALAPPDATA%\llvm-portable\clang+llvm-23.1.2-x86_64-pc-windows-msvc"
if not exist "%LLVM_ROOT%\bin\clang-cl.exe" ( echo ERROR: clang-cl not found under "C:\Program Files\LLVM" or "%LOCALAPPDATA%\llvm-portable" & exit /b 1 )
set "PATH=%LLVM_ROOT%\bin;%PATH%"
set "CV490_ROOT=C:\opencv"
if not exist "%CV490_ROOT%\build\include\opencv2\core.hpp" set "CV490_ROOT=%LOCALAPPDATA%\opencv-portable\opencv"
if not exist "%CV490_ROOT%\build\include\opencv2\core.hpp" ( echo ERROR: OpenCV 4.90 not found under "C:\opencv" or "%LOCALAPPDATA%\opencv-portable\opencv" & exit /b 1 )
set "ROOT=%~dp0"
set "OUT=%ROOT%x64\Cpu"
set "OBJ=%OUT%\obj"
set "OBJT=%OUT%\obj_test"
mkdir "%OUT%" 2>nul
mkdir "%OBJ%" 2>nul
mkdir "%OBJT%" 2>nul
set "CV490_INC=/I %CV490_ROOT%\build\include"
set "CV413_INC=/I C:\Users\iorda\OneDrive\Documents\opencv\Release\include"
set "DEFS=/DNDEBUG /D_CONSOLE /D_CRT_SECURE_NO_WARNINGS"
set "OPTS=/c /O2 /fp:fast /arch:AVX2 /Oi /std:c++17 /EHsc /MD /openmp /Zi"
set "BE_INC=/I %ROOT%ImgReconstruct_backend /I %ROOT%third_party\onnxruntime\include"

echo ============ main objects [opencv 4.90 world] ============
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\CS_encryption.cpp" /Fo"%OBJ%\CS_encryption.obj"
if errorlevel 1 ( echo ERROR CS_encryption & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\crypto_utils.cpp" /Fo"%OBJ%\crypto_utils.obj"
if errorlevel 1 ( echo ERROR crypto_utils & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\cs_wavelet.cpp" /Fo"%OBJ%\cs_wavelet.obj"
if errorlevel 1 ( echo ERROR cs_wavelet & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\helper_functions.cpp" /Fo"%OBJ%\helper_functions.obj"
if errorlevel 1 ( echo ERROR helper_functions & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_decryption.cpp" /Fo"%OBJ%\image_decryption.obj"
if errorlevel 1 ( echo ERROR image_decryption & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_encryption.cpp" /Fo"%OBJ%\image_encryption.obj"
if errorlevel 1 ( echo ERROR image_encryption & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_tiles.cpp" /Fo"%OBJ%\image_tiles.obj"
if errorlevel 1 ( echo ERROR image_tiles & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_preview.cpp" /Fo"%OBJ%\image_preview.obj"
if errorlevel 1 ( echo ERROR image_preview & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\photo_upscaler.cpp" /Fo"%OBJ%\photo_upscaler.obj"
if errorlevel 1 ( echo ERROR photo_upscaler & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\cs_gpu_cpu.cpp" /Fo"%OBJ%\cs_gpu.obj"
if errorlevel 1 ( echo ERROR cs_gpu_cpu & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\ImgReconstruct_backend.cpp" /Fo"%OBJ%\main.obj"
if errorlevel 1 ( echo ERROR main & exit /b 1 )

if "%~1"=="main" goto :linkmain
echo ============ test objects [opencv 4.13 static + contrib] ============
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\CS_encryption.cpp" /Fo"%OBJT%\CS_encryption.obj"
if errorlevel 1 ( echo ERROR CS_encryption_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\crypto_utils.cpp" /Fo"%OBJT%\crypto_utils.obj"
if errorlevel 1 ( echo ERROR crypto_utils_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\cs_wavelet.cpp" /Fo"%OBJT%\cs_wavelet.obj"
if errorlevel 1 ( echo ERROR cs_wavelet_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\helper_functions.cpp" /Fo"%OBJT%\helper_functions.obj"
if errorlevel 1 ( echo ERROR helper_functions_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_decryption.cpp" /Fo"%OBJT%\image_decryption.obj"
if errorlevel 1 ( echo ERROR image_decryption_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_encryption.cpp" /Fo"%OBJT%\image_encryption.obj"
if errorlevel 1 ( echo ERROR image_encryption_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_tiles.cpp" /Fo"%OBJT%\image_tiles.obj"
if errorlevel 1 ( echo ERROR image_tiles_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_preview.cpp" /Fo"%OBJT%\image_preview.obj"
if errorlevel 1 ( echo ERROR image_preview_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\photo_upscaler.cpp" /Fo"%OBJT%\photo_upscaler.obj"
if errorlevel 1 ( echo ERROR photo_upscaler_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\cs_gpu_cpu.cpp" /Fo"%OBJT%\cs_gpu.obj"
if errorlevel 1 ( echo ERROR cs_gpu_cpu_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%tests\test_main.cpp" /Fo"%OBJT%\test_main.obj"
if errorlevel 1 ( echo ERROR test_main & exit /b 1 )

:linkmain
set "SYSLIBS=bcrypt.lib libomp.lib user32.lib gdi32.lib kernel32.lib ole32.lib oleaut32.lib uuid.lib advapi32.lib shell32.lib winspool.lib comdlg32.lib onnxruntime.lib delayimp.lib"

echo [link] ImgReconstruct_backend.exe
link /OUT:"%OUT%\ImgReconstruct_backend.exe" /DEBUG /SUBSYSTEM:CONSOLE /DELAYLOAD:onnxruntime.dll /LIBPATH:%CV490_ROOT%\build\x64\vc16\lib /LIBPATH:%ROOT%third_party\onnxruntime\lib "/LIBPATH:%LLVM_ROOT%\lib" "%OBJ%\main.obj" "%OBJ%\CS_encryption.obj" "%OBJ%\crypto_utils.obj" "%OBJ%\cs_wavelet.obj" "%OBJ%\helper_functions.obj" "%OBJ%\image_decryption.obj" "%OBJ%\image_encryption.obj" "%OBJ%\image_tiles.obj" "%OBJ%\image_preview.obj" "%OBJ%\photo_upscaler.obj" "%OBJ%\cs_gpu.obj" opencv_world490.lib %SYSLIBS%
if errorlevel 1 ( echo ERROR linking exe & exit /b 1 )
if "%~1"=="main" goto :finish

echo [link] cs_tests.exe (static 4.13 set)
set "STATIC413=opencv_core4130.lib opencv_imgproc4130.lib opencv_highgui4130.lib opencv_imgcodecs4130.lib opencv_dnn4130.lib opencv_photo4130.lib opencv_videoio4130.lib opencv_quality4130.lib ippicvmt.lib ippiw.lib ipphal.lib ittnotify.lib libprotobuf.lib libjpeg-turbo.lib libpng.lib libtiff.lib libopenjp2.lib IlmImf.lib zlib.lib ade.lib"
link /OUT:"%OUT%\cs_tests.exe" /DEBUG /SUBSYSTEM:CONSOLE /DELAYLOAD:onnxruntime.dll "/LIBPATH:C:\Users\iorda\OneDrive\Documents\opencv\Release\lib" /LIBPATH:%ROOT%third_party\onnxruntime\lib "/LIBPATH:%LLVM_ROOT%\lib" "%OBJT%\test_main.obj" "%OBJT%\CS_encryption.obj" "%OBJT%\crypto_utils.obj" "%OBJT%\cs_wavelet.obj" "%OBJT%\helper_functions.obj" "%OBJT%\image_decryption.obj" "%OBJT%\image_encryption.obj" "%OBJT%\image_tiles.obj" "%OBJT%\image_preview.obj" "%OBJT%\photo_upscaler.obj" "%OBJT%\cs_gpu.obj" %STATIC413% %SYSLIBS%
if errorlevel 1 ( echo ERROR linking tests & exit /b 1 )

:finish
copy "%LLVM_ROOT%\bin\libomp.dll" "%OUT%\" >nul
copy "%CV490_ROOT%\build\x64\vc16\bin\opencv_world490.dll" "%OUT%\" >nul
echo BUILD OK: %OUT%
