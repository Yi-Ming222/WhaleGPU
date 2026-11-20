@echo off
setlocal
call "C:\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39
if errorlevel 1 exit /b 1
pushd "%~dp0"
if exist "%~1" exit /b 2
mkdir "%~1"
set "INC=-I..\WhaleGPU-Phase1\include -I..\WhaleGPU-Phase34\runs\20261118-092322-b6e99f\source"
nvcc --version > "%~1\compiler.txt"
nvcc -std=c++17 -O3 -arch=sm_120 -lineinfo -Xcompiler /openmp %INC% -gencode arch=compute_120,code=compute_120 bench.cu -o "%~1\bench.exe" > "%~1\build.log" 2>&1
if errorlevel 1 exit /b 1
set "PATH=%CD%\tools;%PATH%"
tools\cuobjdump.exe -sass "%~1\bench.exe" > "%~1\sass.txt"
if errorlevel 1 exit /b 1
tools\cuobjdump.exe -ptx "%~1\bench.exe" > "%~1\ptx.txt"
if errorlevel 1 exit /b 1
set OMP_NUM_THREADS=8
"%~1\bench.exe" "%~1\samples.csv" > "%~1\execution.log" 2>&1
exit /b %errorlevel%
