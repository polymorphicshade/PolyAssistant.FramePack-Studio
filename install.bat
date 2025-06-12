@echo off
echo FramePack-Studio Setup Script
setlocal enabledelayedexpansion

REM Check if Python is installed (basic check)
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH. Please install Python and try again.
    goto end
)

if exist "%cd%/venv" (
echo Error: Virtual Environment already exists. Please run update.bat if you want to update or delete the venv folder first.
goto end
)

REM Check the python version
echo Python versions 3.10-3.12 have been confirmed to work. Other versions are currently not supported. You currently have:
python -V
set /p choice= "Do you want to continue?[Y/N]: "

REM This gets the shortened Python version for later use. e.g. 3.10.13 becomes 310.
for /f "delims=" %%A in ('python -V') do set "pyv=%%A"
for /f "tokens=2 delims= " %%A in ("%pyv%") do (
    set pyv=%%A
)
set pyv=%pyv:.=%
set pyv=%pyv:~0,3%

if "!choice!" == "y" (goto makevenv)
if "!choice!"=="Y" (goto makevenv)

goto end

:makevenv
REM This creates a virtual environment in the folder
echo Creating a Virtual Environment...
python -m venv venv

REM ask Windows for GPU
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Nvidia GPU doesn't exist or drivers installed incorrectly. Please confirm your drivers are installed.
    goto end
)
for /F "tokens=* skip=1" %%n in ('nvidia-smi --query-gpu=name') do set GPU_NAME=%%n && goto checkgpu

:checkgpu
echo Detected %GPU_NAME%
set "GPU_SERIES=%GPU_NAME:*RTX =%"
set "GPU_SERIES=%GPU_SERIES:~0,2%00"

echo Installing torch...

"%cd%/venv/Scripts/pip.exe" install typing-extensions

if !GPU_SERIES! geq 5000 (
	goto torch270
) else (
	goto torch260
)

REM RTX 5000 Series
:torch270
"%cd%/venv/Scripts/pip.exe" install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 

REM Ask if user wants Sage Attention
set choice=
set /p choice= "Do you want to install Sage Attention?(Speeds up generation)[Y/N]: "

if "!choice!" == "y" (goto triton270)
if "!choice!"=="Y" (goto triton270)

goto requirements

:triton270
REM Sage Attention and Triton for Torch 2.7.0
"%cd%/venv/Scripts/pip.exe" install "triton-windows<3.4" 
"%cd%/venv/Scripts/pip.exe" install "https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu128torch2.7.0-cp%pyv%-cp%pyv%-win_amd64.whl"
goto requirements

REM RTX 4000 Series and below
:torch260
"%cd%/venv/Scripts/pip.exe" install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 

REM Ask if user wants Sage Attention
set choice=
echo Do you want to install any of the following? They speed up generation.
echo 1) Sage Attention
echo 2) Flash Attention 
echo 3) BOTH!
echo 4) No
set /p choice= "Do you want to install Sage Attention?(Speeds up generation)[Y/N]: "

set both="N"

if "!choice!" == 1 (goto triton260)
if "!choice!"== 2 (goto flash-attn)
if "!choice!"== 3 (set both="Y"
goto triton260)

goto requirements

:triton260
REM Sage Attention and Triton for Torch 2.6.0
"%cd%/venv/Scripts/pip.exe" install "triton-windows<3.3.0"
"%cd%/venv/Scripts/pip.exe" install  https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu126torch2.6.0-cp%pyv%-cp%pyv%-win_amd64.whl

if %both% == "Y" (goto flash-attn)

goto requirements

:flash-attn
REM Install Flash Attention
"%cd%/venv/Scripts/pip.exe" install flash-attn

:requirements
echo Installing remaining required packages through pip...
REM This assumes there's a requirements.txt file in the root
"%cd%/venv/Scripts/pip.exe" install -r requirements.txt 

REM Check if pip installation was successful
if %errorlevel% neq 0 (
    echo Warning: Failed to install dependencies. You may need to install them manually.
    goto end
)

echo Setup complete.

:end
echo Exiting setup script.
pause
