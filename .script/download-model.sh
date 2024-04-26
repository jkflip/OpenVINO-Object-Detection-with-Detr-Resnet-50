#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
work_dir="$SCRIPT_DIR"/../
model_dir="$work_dir"/model

echo "Downloading OpenVINO models from model.lst online"
source "${work_dir}"/venv/bin/activate

cd "${model_dir}"
omz_downloader --list model.lst --output_dir "${model_dir}"
echo "Done"
