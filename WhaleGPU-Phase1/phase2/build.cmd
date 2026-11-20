@echo off
setlocal
if "%~1"=="" exit /b 2
call "C:\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39
if errorlevel 1 exit /b 1
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"
pushd "%~dp0"
if not exist "%~1" mkdir "%~1"
nvcc --version
nvcc -arch=sm_120 -std=c++17 -O3 -lineinfo -I..\include -Isrc src\bench.cu ..\src\reduction.cu -o "%~1\whalegpu_bench.exe"
set "BUILD_RESULT=%ERRORLEVEL%"
popd
exit /b %BUILD_RESULT%
