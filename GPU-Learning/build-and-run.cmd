@echo off
setlocal
call "C:\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"
pushd "%~dp0"
if not exist build mkdir build
nvcc -arch=sm_120 -std=c++17 "01-vector-add\main.cu" -o "build\vector_add.exe"
if errorlevel 1 goto failed
"build\vector_add.exe"
if errorlevel 1 goto failed
popd
exit /b 0
:failed
popd
exit /b 1
