@echo off
echo FramePack-Studio Setup Script

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
echo The recommended version of Python is 3.10. You currently have:
python --version 
echo Anything above 3.11 is not confirmed to fully work
echo Do you want to continue? 
echo 1) Yes 
echo 2) No
set /p choice= "Enter your choice: "

if %choice% == 1 (

REM This creates a virtual environment in the folder
echo Creating a Virtual Environment
python -m venv venv

) else ( 
	goto end 
)

REM Ask for GPU and install torch version accordingly
echo What GPU do you have?
echo 1) RTX 4000 series and below
echo 2) RTX 5000 series
set /p choice2= "Enter your choice: "

if %choice2% == 1 (

echo Installing torch...
"%cd%/venv/Scripts/pip.exe" install typing-extensions
"%cd%/venv/Scripts/pip.exe" install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --use-pep517

) else if %choice2% == 2 (

echo Installing torch...
"%cd%/venv/Scripts/pip.exe" install typing-extensions
"%cd%/venv/Scripts/pip.exe" install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --use-pep517

) else (

echo You entered an invalid choice
goto end
)


echo Installing dependencies using pip...
REM This assumes there's a requirements.txt file in the root
"%cd%/venv/Scripts/pip.exe" install -r requirements.txt --use-pep517

REM Check if pip installation was successful
if %errorlevel% neq 0 (
    echo Warning: Failed to install dependencies. You may need to install them manually.
    goto end
)

echo Setup complete.

:end
echo Exiting setup script.
pause
