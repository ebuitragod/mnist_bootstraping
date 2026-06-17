#! /bin/bash

git config --global --add safe.directory /workspaces/apps

# Fix permissions for SSH agent socket
sudo chmod 666 /run/host-services/ssh-auth.sock && sudo usermod -a -G $(stat -c '%G' /run/host-services/ssh-auth.sock) vscode

# Ensure no venv exists -> creates horrible to trace issues with UV
rm -rf /workspaces/apps/.venv
echo "=============== postStartCommand ran successfully. ==============="
