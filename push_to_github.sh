#!/bin/bash
# ── Showstopper: Push to GitHub ──
# Run this script AFTER creating the repo at https://github.com/new
# Repo name: Showstopper | Public | No README/gitignore/license

echo "🚀 Adding GitHub remote..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/Mohamed-Abdelwahab11/Showstopper.git

echo "📤 Pushing to GitHub..."
git push -u origin main

echo "✅ Done! Visit: https://github.com/Mohamed-Abdelwahab11/Showstopper"
