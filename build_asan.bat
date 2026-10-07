@echo off
REM ASAN diagnosis build: main exe only, same sources/flags as build_clang.bat
REM plus /fsanitize=address (+use-after-scope). Output: x64\ClangAsan.
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )
set "LLVM_ROOT=C:\Program Files\LLVM"
if not exist "%LLVM_ROOT%\bin\clang-cl.exe" set "LLVM_ROOT=%LOCALAPPDATA%\llvm-portable\clang+llvm-23.1.2-x86_64-pc-windows-msvc"
if not exist "%LLVM_ROOT%\bin\clang-cl.exe" ( echo ERROR: clang-cl not found under "C:\Program Files\LLVM" or "%LOCALAPPDATA%\llvm-portable" & exit /b 1 )
set "PATH=%LLVM_ROOT%\bin;%PATH%"
set "ROOT=%~dp0"
set "OUT=%ROOT%x64\ClangAsan"
set "OBJ=%OUT%\obj"
mkdir "%OUT%" 2>nul
mkdir "%OBJ%" 2>nul
set "CV490_INC=/I C:\opencv\build\include"
set "DEFS=/DNDEBUG /D_CONSOLE /D_CRT_SECURE_NO_WARNINGS"
set "OPTS=/c /O2 /fp:fast /arch:AVX2 /Oi /std:c++17 /EHsc /MD /openmp /Zi /fsanitize=address /D_DISABLE_STL_ANNOTATION"
set "BE_INC=/I %ROOT%ImgReconstruct_backend /I %ROOT%third_party\onnxruntime\include"
set "SYSLIBS=bcrypt.lib libomp.lib user32.lib gdi32.lib kernel32.lib ole32.lib oleaut32.lib uuid.lib advapi32.lib shell32.lib winspool.lib comdlg32.lib onnxruntime.lib delayimp.lib amdhip64.lib hipfft.lib"

if "%~1"=="linkonly" goto :link
if "%~1"=="gpuobj" goto :gpu

echo ============ main objects [ASAN] ============
for %%f in (CS_encryption crypto_utils cs_dict cs_wavelet helper_functions image_decryption image_encryption image_tiles image_preview photo_upscaler) do (
  clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\%%f.cpp" /Fo"%OBJ%\%%f.obj"
  if errorlevel 1 ( echo ERROR %%f & exit /b 1 )
)
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\ImgReconstruct_backend.cpp" /Fo"%OBJ%\main.obj"
if errorlevel 1 ( echo ERROR main & exit /b 1 )

copy "%ROOT%x64\Clang\obj\cs_gpu.obj" "%OBJ%\cs_gpu.obj" >nul
if not exist "%OBJ%\cs_gpu.obj" ( echo ERROR cs_gpu.obj missing from normal build & exit /b 1 )

echo [link] ImgReconstruct_backend.exe [ASAN]
:link
link /OUT:"%OUT%\ImgReconstruct_backend.exe" /DEBUG /SUBSYSTEM:CONSOLE /DELAYLOAD:onnxruntime.dll /LIBPATH:C:\opencv\build\x64\vc16\lib /LIBPATH:%ROOT%third_party\onnxruntime\lib "/LIBPATH:C:\Program Files\AMD\ROCm\7.2\lib" "/LIBPATH:%LLVM_ROOT%\lib" "/LIBPATH:%LLVM_ROOT%\lib\clang\23\lib\windows" /fsanitize=address "%OBJ%\main.obj" "%OBJ%\CS_encryption.obj" "%OBJ%\crypto_utils.obj" "%OBJ%\cs_dict.obj" "%OBJ%\cs_wavelet.obj" "%OBJ%\helper_functions.obj" "%OBJ%\image_decryption.obj" "%OBJ%\image_encryption.obj" "%OBJ%\image_tiles.obj" "%OBJ%\image_preview.obj" "%OBJ%\photo_upscaler.obj" "%OBJ%\cs_gpu.obj" opencv_world490.lib clang_rt.asan_dynamic-x86_64.lib clang_rt.asan_dynamic_runtime_thunk-x86_64.lib %SYSLIBS%
if errorlevel 1 ( echo ERROR linking exe & exit /b 1 )

copy "%LLVM_ROOT%\lib\clang\23\lib\windows\clang_rt.asan_dynamic-x86_64.dll" "%OUT%\" >nul
copy "%LLVM_ROOT%\bin\libomp.dll" "%OUT%\" >nul
copy "C:\Program Files\AMD\ROCm\7.2\bin\amdhip64_7.dll" "%OUT%\" >nul
copy "C:\Program Files\AMD\ROCm\7.2\bin\hipfft.dll" "%OUT%\" >nul
copy "C:\Program Files\AMD\ROCm\7.2\bin\rocfft.dll" "%OUT%\" >nul
copy "%ROOT%third_party\onnxruntime\lib\onnxruntime.dll" "%OUT%\" >nul
echo BUILD OK: %OUT%
goto :eof

:gpu
echo [hipcc+ASAN] cs_gpu.hip
call "C:\Program Files\AMD\ROCm\7.2\bin\hipcc.bat" --offload-arch=gfx1100 -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS -fms-runtime-lib=dll -fsanitize=address -D_DISABLE_STL_ANNOTATION -c "%ROOT%ImgReconstruct_backend\cs_gpu.hip" -o "%OBJ%\cs_gpu.obj"
if errorlevel 1 ( echo ERROR cs_gpu.hip & exit /b 1 )
echo GPU OBJ OK
