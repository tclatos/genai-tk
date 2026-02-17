#!/bin/bash
# Setup script for Deer-flow integration
# This script helps set up the DEER_FLOW_PATH environment variable and install deer-flow

set -e

echo "🦌 Deer-flow Setup Script"
echo "========================"
echo

# Check if DEER_FLOW_PATH is already set
if [ -n "$DEER_FLOW_PATH" ]; then
    echo "✅ DEER_FLOW_PATH is already set to: $DEER_FLOW_PATH"
    DEER_FLOW_ROOT="$DEER_FLOW_PATH"
else
    # Prompt for installation location
    echo "Where would you like to install deer-flow?"
    echo "  1. ~/ext_prj/deer-flow (recommended)"
    echo "  2. ./ext/deer-flow (in current project)"
    echo "  3. Custom path"
    read -p "Enter choice [1-3]: " choice

    case $choice in
        1)
            DEER_FLOW_ROOT="$HOME/ext_prj/deer-flow"
            ;;
        2)
            DEER_FLOW_ROOT="$(pwd)/ext/deer-flow"
            ;;
        3)
            read -p "Enter custom path: " DEER_FLOW_ROOT
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

echo
echo "Deer-flow will be installed to: $DEER_FLOW_ROOT"
echo

# Clone deer-flow if it doesn't exist
if [ ! -d "$DEER_FLOW_ROOT" ]; then
    echo "📥 Cloning deer-flow repository..."
    mkdir -p "$(dirname "$DEER_FLOW_ROOT")"
    git clone --depth 1 https://github.com/bytedance/deer-flow.git "$DEER_FLOW_ROOT"
    echo "✅ Cloned deer-flow"
else
    echo "✅ Deer-flow repository already exists"
fi

# Verify backend exists
if [ ! -f "$DEER_FLOW_ROOT/backend/src/__init__.py" ]; then
    echo "❌ Error: Invalid deer-flow installation at $DEER_FLOW_ROOT"
    echo "   Backend directory or __init__.py not found"
    exit 1
fi

# Install deer-flow package
echo
echo "📦 Installing deer-flow package..."
uv pip install -e "$DEER_FLOW_ROOT/backend"
echo "✅ Deer-flow package installed"

# Add to shell profile
echo
echo "🔧 Setting up environment variable..."

# Detect shell
SHELL_NAME=$(basename "$SHELL")
case "$SHELL_NAME" in
    bash)
        PROFILE="$HOME/.bashrc"
        ;;
    zsh)
        PROFILE="$HOME/.zshrc"
        ;;
    fish)
        PROFILE="$HOME/.config/fish/config.fish"
        ;;
    *)
        PROFILE="$HOME/.profile"
        ;;
esac

# Check if already in profile
if grep -q "DEER_FLOW_PATH" "$PROFILE" 2>/dev/null; then
    echo "⚠️  DEER_FLOW_PATH already exists in $PROFILE"
    echo "   Current value: $(grep DEER_FLOW_PATH "$PROFILE" | head -n 1)"
    read -p "   Update it? [y/N]: " update
    if [ "$update" = "y" ] || [ "$update" = "Y" ]; then
        # Remove old entries
        sed -i.bak '/DEER_FLOW_PATH/d' "$PROFILE"
        echo "export DEER_FLOW_PATH=\"$DEER_FLOW_ROOT\"" >> "$PROFILE"
        echo "✅ Updated DEER_FLOW_PATH in $PROFILE"
    fi
else
    echo "export DEER_FLOW_PATH=\"$DEER_FLOW_ROOT\"" >> "$PROFILE"
    echo "✅ Added DEER_FLOW_PATH to $PROFILE"
fi

# Set for current session
export DEER_FLOW_PATH="$DEER_FLOW_ROOT"

# Test installation
echo
echo "🧪 Testing installation..."
if python -c "from genai_tk.extra.agents.deer_flow._path_setup import get_deer_flow_backend_path; get_deer_flow_backend_path()" 2>/dev/null; then
    echo "✅ Installation test passed!"
else
    echo "❌ Installation test failed"
    exit 1
fi

echo
echo "=========================================="
echo "✅ Setup complete!"
echo
echo "To use deer-flow in new terminal sessions:"
echo "  source $PROFILE"
echo
echo "Or for immediate use in this session:"
echo "  export DEER_FLOW_PATH=\"$DEER_FLOW_ROOT\""
echo
echo "Test with:"
echo "  cli agents deerflow --list"
echo "=========================================="
