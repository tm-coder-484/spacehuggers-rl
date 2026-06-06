#!/usr/bin/env bash
set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Installing Playwright + Chromium..."
playwright install chromium
playwright install-deps chromium

echo "==> Cloning SpaceHuggers..."
if [ ! -d "SpaceHuggers-main" ]; then
    git clone https://github.com/KilledByAPixel/SpaceHuggers.git SpaceHuggers-main
fi

echo "==> Creating model/log directories..."
mkdir -p game_models game_logs

echo ""
echo "Setup complete!"
echo "  Train:       python train_game.py --forever --envs 3"
echo "  TensorBoard: tensorboard --logdir game_logs"
