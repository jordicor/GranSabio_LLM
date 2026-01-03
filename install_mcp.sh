#!/bin/bash
#
# Gran Sabio LLM MCP Server Installer
# Registers the MCP server with Claude Code using the correct absolute path.
#
# Usage:
#   ./install_mcp.sh              # Install with defaults
#   ./install_mcp.sh --uninstall  # Remove the MCP server
#

set -e

# Get the directory where this script is located (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_SERVER_PATH="$SCRIPT_DIR/mcp_server/gransabio_mcp_server.py"
MCP_NAME="gransabio-llm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() { echo -e "${GREEN}$1${NC}"; }
print_warning() { echo -e "${YELLOW}$1${NC}"; }
print_error() { echo -e "${RED}$1${NC}"; }

# Check if claude CLI is available
if ! command -v claude &> /dev/null; then
    print_error "Error: 'claude' CLI not found in PATH"
    echo "Please install Claude Code first: https://claude.ai/code"
    exit 1
fi

# Check if MCP server file exists
if [ ! -f "$MCP_SERVER_PATH" ]; then
    print_error "Error: MCP server not found at: $MCP_SERVER_PATH"
    exit 1
fi

# Handle uninstall
if [ "$1" = "--uninstall" ] || [ "$1" = "-u" ]; then
    echo "Removing Gran Sabio LLM MCP server..."
    if claude mcp remove "$MCP_NAME" 2>/dev/null; then
        print_success "MCP server '$MCP_NAME' removed successfully."
    else
        print_warning "MCP server '$MCP_NAME' was not registered or already removed."
    fi
    exit 0
fi

# Check for Python
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Error: Python not found. Please install Python 3.10+"
    exit 1
fi

echo "==================================="
echo "Gran Sabio LLM MCP Server Installer"
echo "==================================="
echo ""
echo "This will register the Gran Sabio LLM MCP server with Claude Code."
echo ""
echo "Server path: $MCP_SERVER_PATH"
echo "Python: $PYTHON_CMD"
echo ""

# Check if MCP dependencies are installed
echo "Checking MCP dependencies..."
if ! $PYTHON_CMD -c "import mcp" 2>/dev/null; then
    print_warning "MCP SDK not found. Installing..."
    pip install -r "$SCRIPT_DIR/mcp_server/requirements.txt"
    print_success "Dependencies installed."
fi

# Register the MCP server
echo ""
echo "Registering MCP server with Claude Code..."

# Build the command
# Using absolute path ensures it works regardless of working directory
claude mcp add "$MCP_NAME" -- "$PYTHON_CMD" "$MCP_SERVER_PATH"

echo ""
print_success "Installation complete!"
echo ""
echo "The MCP server is now available in Claude Code."
echo "Available tools:"
echo "  - gransabio_analyze_code    : Analyze code for bugs and security issues"
echo "  - gransabio_review_fix      : Review proposed code fixes"
echo "  - gransabio_generate_with_qa: Generate content with multi-model QA"
echo "  - gransabio_check_health    : Check API connectivity"
echo "  - gransabio_list_models     : List available AI models"
echo "  - gransabio_get_config      : View current configuration"
echo ""
echo "Make sure Gran Sabio LLM API is running: python main.py"
echo ""
echo "To uninstall: ./install_mcp.sh --uninstall"
