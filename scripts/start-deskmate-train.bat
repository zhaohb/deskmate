@echo off
REM ============================================================================
REM  start-deskmate-train.bat — launch DeskMate's UI from the GPU-training env
REM  with the Intel oneAPI + MSVC toolchain loaded, so in-process LoRA training
REM  can JIT-compile Unsloth/Triton SYCL kernels on the Intel Arc iGPU (XPU).
REM
REM  Fully auto-detecting / portable — no hard-coded versions or user paths:
REM    * MSVC      : located via vswhere (any VS edition / Build Tools).
REM    * oneAPI    : newest "compiler\<ver>" found under the oneAPI install.
REM    * train env : `mamba/conda run -n deskmate_train` (env resolved by name).
REM
REM  Why this wrapper exists:
REM    Training runs IN-PROCESS inside the UI server (engine/api.py), so the UI
REM    must run from the deskmate_train env (torch+xpu + Unsloth). Triton-XPU
REM    JIT-compiles kernels with Intel's `icx`, a clang-cl driver that needs
REM    BOTH the MSVC standard library (climits, vcruntime, …) AND the oneAPI
REM    SYCL headers/libs on INCLUDE/LIB.
REM
REM  Override any auto-detection with env vars before calling:
REM    set "DESKMATE_VCVARS=...\vcvarsall.bat"
REM    set "DESKMATE_ONEAPI=...\Intel\oneAPI"
REM    set "DESKMATE_TRAIN_ENV=deskmate_train"
REM ============================================================================
setlocal EnableDelayedExpansion

REM --- 1) MSVC — locate vcvarsall.bat via vswhere ------------------------------
set "VCVARS=%DESKMATE_VCVARS%"
if not defined VCVARS (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if exist "!VSWHERE!" (
    for /f "usebackq delims=" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
    if defined VSROOT set "VCVARS=!VSROOT!\VC\Auxiliary\Build\vcvarsall.bat"
  )
)
if not defined VCVARS goto :no_msvc
if not exist "%VCVARS%" goto :no_msvc
call "%VCVARS%" x64 >nul 2>&1
echo [ok] MSVC x64 environment loaded.
goto :oneapi
:no_msvc
echo [ERROR] Could not locate vcvarsall.bat. Install Visual Studio Build Tools
echo         with the "Desktop development with C++" workload, or set DESKMATE_VCVARS.
exit /b 1

REM --- 2) Intel oneAPI DPC++/C++ compiler (icx) for SYCL kernels ---------------
:oneapi
set "ONEAPI_HOME=%DESKMATE_ONEAPI%"
if not defined ONEAPI_HOME set "ONEAPI_HOME=%ProgramFiles(x86)%\Intel\oneAPI"
if not defined ONEAPI_HOME set "ONEAPI_HOME=%ProgramFiles%\Intel\oneAPI"
set "CMPLR="
if exist "%ONEAPI_HOME%\compiler\" (
  REM newest version dir (reverse-sorted) that actually has icx.exe
  for /f "delims=" %%D in ('dir /b /ad /o-n "%ONEAPI_HOME%\compiler" 2^>nul') do (
    if not defined CMPLR if exist "%ONEAPI_HOME%\compiler\%%D\bin\icx.exe" set "CMPLR=%ONEAPI_HOME%\compiler\%%D"
  )
)
if not defined CMPLR goto :no_oneapi
set "PATH=%CMPLR%\bin;%PATH%"
set "INCLUDE=%CMPLR%\include;%INCLUDE%"
set "LIB=%CMPLR%\lib;%LIB%"
set "CXX=%CMPLR%\bin\icx.exe"
set "CC=%CMPLR%\bin\icx.exe"
echo [ok] Intel oneAPI compiler loaded: %CMPLR% (CXX=icx).
goto :launch
:no_oneapi
echo [ERROR] icx.exe not found under "%ONEAPI_HOME%\compiler\*\bin".
echo         Install the Intel oneAPI Base Toolkit (DPC++/C++ Compiler),
echo         or set DESKMATE_ONEAPI to the oneAPI install root.
exit /b 1

REM --- 3) Launch DeskMate UI from the GPU-training env -------------------------
:launch
set "TRAIN_ENV=%DESKMATE_TRAIN_ENV%"
if not defined TRAIN_ENV set "TRAIN_ENV=deskmate_train"
where mamba >nul 2>&1 && (set "CONDA=mamba") || (set "CONDA=conda")
echo [ok] Starting DeskMate UI (%CONDA% run -n %TRAIN_ENV%, XPU training enabled)...
echo.
%CONDA% run --no-capture-output -n %TRAIN_ENV% python -m deskmate ui %*
endlocal
