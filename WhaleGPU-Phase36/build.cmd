@echo off
setlocal
call "C:\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39
if errorlevel 1 exit /b 1
pushd "%~dp0"
if "%~1"=="" exit /b 2
if exist "%~1" exit /b 2
mkdir "%~1"
nvcc -std=c++17 -O3 -arch=sm_120 -gencode arch=compute_120,code=compute_120 -lineinfo -Xcompiler /openmp bench.cu -o "%~1\bench.exe" > "%~1\build.log" 2>&1
if errorlevel 1 exit /b 1
set "PATH=%CD%\..\WhaleGPU-Phase35\tools;%PATH%"
cuobjdump -sass "%~1\bench.exe" > "%~1\sass.txt"
if errorlevel 1 exit /b 1
cuobjdump -ptx "%~1\bench.exe" > "%~1\ptx.txt"
if errorlevel 1 exit /b 1
set OMP_NUM_THREADS=8
"%~1\bench.exe" "%~1\first" 36 > "%~1\first.log" 2>&1
if errorlevel 1 exit /b 1
"%~1\bench.exe" "%~1\repeat" 360 > "%~1\repeat.log" 2>&1
exit /b %errorlevel%
