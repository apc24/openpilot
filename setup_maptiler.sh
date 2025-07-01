#!/usr/bin/env bash

# MapTiler setup script for openpilot UI
# This script sets up the MapTiler token for map rendering

echo "Setting up MapTiler for openpilot UI..."

# Check if MAPTILER_TOKEN is already set
if [ -z "$MAPTILER_TOKEN" ]; then
  echo "MAPTILER_TOKEN environment variable is not set."
  echo "Please obtain a MapTiler API key from https://www.maptiler.com/"
  echo ""
  echo "Steps to get your API key:"
  echo "1. Go to https://www.maptiler.com/"
  echo "2. Sign up for a free account"
  echo "3. Go to your dashboard and copy your API key"
  echo "4. Set it using: export MAPTILER_TOKEN=YOUR_MAPTILER_KEY"
  echo "5. Or add it to your ~/.bashrc file for persistence"
  echo ""
  echo "Note: The token 'YOUR_MAPTILER_KEY' in this script is just an example!"
  exit 1
fi

echo "MapTiler token found: ${MAPTILER_TOKEN:0:8}..."

# Test the token validity
echo "Testing MapTiler token validity..."
response=$(curl -s -o /dev/null -w "%{http_code}" "https://api.maptiler.com/maps/basic/style.json?key=$MAPTILER_TOKEN")

if [ "$response" = "200" ]; then
  echo "✓ MapTiler token is valid!"
elif [ "$response" = "403" ]; then
  echo "✗ MapTiler token is invalid or unauthorized (HTTP 403)"
  echo "Please check your API key at https://cloud.maptiler.com/account/keys/"
  exit 1
elif [ "$response" = "401" ]; then
  echo "✗ MapTiler token authentication failed (HTTP 401)"
  echo "Please verify your API key is correct"
  exit 1
else
  echo "⚠ Unexpected response from MapTiler API (HTTP $response)"
  echo "Proceeding anyway..."
fi

echo "MapTiler setup complete!"
echo ""
echo "To use MapTiler in openpilot UI:"
echo "1. Make sure MAPTILER_TOKEN is set in your environment"
echo "2. Rebuild the UI components if necessary"
echo "3. Start openpilot normally"