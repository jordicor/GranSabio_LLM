@echo off
setlocal enabledelayedexpansion

REM ===================================
REM Gran Sabio LLM MCP Server Installer
REM Registers the MCP server with Claude Code using the correct absolute path.
REM
REM Usage:
REM   install_mcp.bat              - Install with defaults
REM   install_mcp.bat --uninstall  - Remove the MCP server
REM ===================================

set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "MCP_SERVER_PATH=%SCRIPT_DIR%\mcp_server\gransabio_mcp_server.py"
set "MCP_NAME=gransabio-llm"

REM Check for uninstall flag
if "%~1"=="--uninstall" goto :uninstall
if "%~1"=="-u" goto :uninstall

REM Check if claude CLI is available
where claude >nul 2>&1
if errorlevel 1 (
    echo Error: 'claude' CLI not found in PATH
    echo Please install Claude Code first: https://claude.ai/code
    exit /b 1
)

REM Check if MCP server file exists
if not exist "%MCP_SERVER_PATH%" (
    echo Error: MCP server not found at: %MCP_SERVER_PATH%
    exit /b 1
)

REM Check for Python
set "PYTHON_CMD="
where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
) else (
    where python3 >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=python3"
    )
)

if "%PYTHON_CMD%"=="" (
    echo Error: Python not found. Please install Python 3.10+
    exit /b 1
)

echo ===================================
echo Gran Sabio LLM MCP Server Installer
echo ===================================
echo.
echo This will register the Gran Sabio LLM MCP server with Claude Code.
echo.
echo Server path: %MCP_SERVER_PATH%
echo Python: %PYTHON_CMD%
echo.

REM Check if MCP dependencies are installed
echo Checking MCP dependencies...
%PYTHON_CMD% -c "import mcp" >nul 2>&1
if errorlevel 1 (
    echo MCP SDK not found. Installing...
    pip install -r "%SCRIPT_DIR%\mcp_server\requirements.txt"
    echo Dependencies installed.
)

echo.
echo Registering MCP server with Claude Code...

REM Register the MCP server with absolute path
claude mcp add %MCP_NAME% -- %PYTHON_CMD% "%MCP_SERVER_PATH%"

if errorlevel 1 (
    echo.
    echo Error: Failed to register MCP server.
    exit /b 1
)

echo.
echo Installation complete!
echo.
echo The MCP server is now available in Claude Code.
echo Available tools:
echo   - gransabio_analyze_code    : Analyze code for bugs and security issues
echo   - gransabio_review_fix      : Review proposed code fixes
echo   - gransabio_generate_with_qa: Generate content with multi-model QA
echo   - gransabio_check_health    : Check API connectivity
echo   - gransabio_list_models     : List available AI models
echo   - gransabio_get_config      : View current configuration
echo.
echo Make sure Gran Sabio LLM API is running: python main.py
echo.
echo To uninstall: install_mcp.bat --uninstall
goto :eof

:uninstall
echo Removing Gran Sabio LLM MCP server...
claude mcp remove %MCP_NAME% >nul 2>&1
if errorlevel 1 (
    echo MCP server '%MCP_NAME%' was not registered or already removed.
) else (
    echo MCP server '%MCP_NAME%' removed successfully.
)
goto :eof
