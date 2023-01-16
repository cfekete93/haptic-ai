#!/bin/bash

set -e

export FLASK_APP=haptic_ai.py
export FLASK_ENV=development # Deprecated! Use below but find proper usage
#export FLASK_DEBUG=development

cd "${HOME}/Projects/haptic/haptic-ai"

flask run
