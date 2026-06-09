@echo off
REM ============================================================================
REM  setup-intel-xpu.bat — one-time, reproducible setup for Intel Arc / iGPU
REM  LoRA training (Unsloth on PyTorch-XPU). Idempotent: safe to re-run.
REM
REM  Prerequisites you install yourself (links in docs/16-learning-training.md):
REM    1. mamba/conda env `deskmate_train` (Python 3.10) with deskmate installed
REM    2. Intel oneAPI Base Toolkit  (provides icpx, the SYCL compiler)
REM    3. Visual Studio Build Tools, "Desktop development with C++" workload
REM    4. Latest Intel GPU driver
REM
REM  This script does the two non-obvious steps that the toolkits DON'T do:
REM    A. installs the Level-Zero SDK *headers* (level_zero/ze_api.h) into the
REM       env — the Base Toolkit ships only the ze_loader.dll runtime, not the
REM       SDK headers Triton needs to JIT-compile XPU kernels.
REM    B. (re)installs torch 2.10.0+xpu so torch.xpu.is_available() is True.
REM ============================================================================
setlocal EnableDelayedExpansion

set "TRAIN_ENV=%DESKMATE_TRAIN_ENV%"
if not defined TRAIN_ENV set "TRAIN_ENV=deskmate_train"
where mamba >nul 2>&1 && (set "CONDA=mamba") || (set "CONDA=conda")

REM --- resolve DeskMate's data dir for the DEDICATED Level-Zero SDK ------------
REM We install the L0 SDK into a clean, dedicated dir (~/.deskmate/level-zero-sdk)
REM rather than the conda env's Library/include. Reason: Triton prepends
REM <ZE_PATH>/include to the SYCL compile, and the conda dpcpp_impl drops an
REM INCOMPLETE sycl/ header tree into Library/include that would then shadow the
REM system oneAPI SYCL headers and break the build (__spirv_* errors). A dir with
REM only level_zero/ avoids the clash. Override the env name with DESKMATE_TRAIN_ENV.
for /f "delims=" %%P in ('%CONDA% run -n %TRAIN_ENV% python -c "import sys;print(sys.prefix)"') do set "ENV_ROOT=%%P"
if not defined ENV_ROOT (
  echo [ERROR] Could not resolve env "%TRAIN_ENV%". Create it first ^(see docs/16^).
  exit /b 1
)
set "DESKMATE_DATA=%DESKMATE_HOME%"
if not defined DESKMATE_DATA set "DESKMATE_DATA=%USERPROFILE%\.deskmate"
set "ZEROOT=%DESKMATE_DATA%\level-zero-sdk"
echo [info] env=%TRAIN_ENV%  Level-Zero SDK target=%ZEROOT%

REM --- A) Level-Zero SDK headers (dedicated clean dir) -------------------------
if exist "%ZEROOT%\include\level_zero\ze_api.h" (
  echo [ok] Level-Zero SDK headers already present.
) else (
  REM Pin a known-good version; override with DESKMATE_LZ_VER if needed.
  set "LZVER=%DESKMATE_LZ_VER%"
  if not defined LZVER set "LZVER=1.29.0"
  set "LZURL=https://github.com/oneapi-src/level-zero/releases/download/v!LZVER!/level-zero-win-sdk-!LZVER!.zip"
  echo [info] Downloading + installing Level-Zero Windows SDK !LZVER! ...
  echo        %LZURL%
  REM Download to a temp dir, extract, copy ONLY level_zero/ headers + libs into
  REM the dedicated dir, then delete the temp dir — nothing is kept in the repo.
  powershell -NoProfile -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; $t=Join-Path $env:TEMP ('lzsdk_'+[guid]::NewGuid()); New-Item -ItemType Directory -Force $t | Out-Null; $zip=Join-Path $t 'sdk.zip'; try { Invoke-WebRequest -UseBasicParsing -Uri '!LZURL!' -OutFile $zip; Expand-Archive -Force $zip $t; New-Item -ItemType Directory -Force '%ZEROOT%\include','%ZEROOT%\lib' | Out-Null; Copy-Item -Recurse -Force (Join-Path $t 'include\level_zero') '%ZEROOT%\include\'; Copy-Item -Force (Join-Path $t 'lib\*.lib') '%ZEROOT%\lib\' } finally { Remove-Item -Recurse -Force $t -ErrorAction SilentlyContinue }"
  if exist "%ZEROOT%\include\level_zero\ze_api.h" (
    echo [ok] Level-Zero SDK headers installed to %ZEROOT%.
  ) else (
    echo [ERROR] Level-Zero SDK header install failed.
    exit /b 1
  )
)

REM --- B) torch 2.10.0+xpu -----------------------------------------------------
echo [info] Verifying torch XPU ...
%CONDA% run --no-capture-output -n %TRAIN_ENV% python -c "import torch,sys; sys.exit(0 if (torch.__version__.endswith('+xpu') and torch.xpu.is_available()) else 1)"
if errorlevel 1 (
  echo [info] Installing torch 2.10.0+xpu ...
  %CONDA% run --no-capture-output -n %TRAIN_ENV% pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/xpu
) else (
  echo [ok] torch+xpu present and XPU available.
)

echo.
echo [done] Intel XPU training environment is ready.
echo        Launch DeskMate with:  scripts\start-deskmate-train.bat
endlocal
