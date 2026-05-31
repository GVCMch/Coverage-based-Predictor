#!/bin/bash

echo "=== Step 1: Initialize Git ==="
git init

echo ""
echo "=== Step 2: Configure Git (replace with your info) ==="
echo "Run these commands manually:"
echo "  git config user.name \"Your Name\""
echo "  git config user.email \"your.email@example.com\""
echo ""
read -p "Press Enter after configuring..."

echo ""
echo "=== Step 3: Add files ==="
git add .

echo ""
echo "=== Step 4: Check status ==="
git status

echo ""
echo "=== Step 5: Commit ==="
git commit -m "Initial commit: Coverage-based Predictor"

echo ""
echo "=== Step 6: Add remote (replace YOUR_USERNAME) ==="
echo "Run this command manually:"
echo "  git remote add origin https://github.com/YOUR_USERNAME/Coverage-based-Predictor.git"
echo ""
read -p "Press Enter after adding remote..."

echo ""
echo "=== Step 7: Push to GitHub ==="
git branch -M main
git push -u origin main

echo ""
echo "✓ Done! Your project is now on GitHub"
