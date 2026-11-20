@echo off
setlocal
call "C:\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39
if errorlevel 1 exit /b 1
pushd "%~dp0"
if "%~1"=="" exit /b 2
if exist "%~1" exit /b 2
mkdir "%~1"
copy /y *.cuh "%~1" >nul
copy /y bench.cu "%~1" >nul
nvcc -arch=sm_120 -std=c++17 -O3 -lineinfo bench.cu -o "%~1\softmax.exe" > "%~1\build.log" 2>&1
if errorlevel 1 exit /b 1
"%~1\softmax.exe" validate "%~1\validation.csv" 41
exit /b %errorlevel%
