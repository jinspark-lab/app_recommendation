#!/bin/bash
set -e

echo "Installing Azure CLI..."
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Get uv path
UV_PATH=$(which uv)

echo "Installing Microsoft Agent Framework..."
sudo -E $UV_PATH pip install --system azure-ai-projects azure-identity

echo "Installing Flask and dependencies..."
sudo -E $UV_PATH pip install --system flask gunicorn

echo "Installing Python packages..."
if [ -f "requirements.txt" ]; then
    sudo -E $UV_PATH pip install --system -r requirements.txt
fi

if [ -f "pyproject.toml" ]; then
    sudo -E $UV_PATH pip install --system -e .
fi

echo "Setup complete!"
