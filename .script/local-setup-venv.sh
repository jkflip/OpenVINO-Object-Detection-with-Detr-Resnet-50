#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

if [ -d "${SCRIPT_DIR}"/../venv ]; then
    echo "Venv already exist"
    exit 0
fi

python3 -m venv "${SCRIPT_DIR}"/../venv
source "${SCRIPT_DIR}"/../venv/bin/activate
pip3 install --upgrade pip
pip3 install -r "${SCRIPT_DIR}"/../requirements.txt

echo "Done"
