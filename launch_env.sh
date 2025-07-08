#!/usr/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="9.6"
fi

export STAGING_ROOT="/data/safe_staging"

# 実行中のスクリプトがsourceされた場合でも正しいパスを取得
script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# setAPIKey.shの相対パス
setAPIKey_script="$script_dir/setAPIKey.sh"

# ファイルが存在するか確認
if [ -f "$setAPIKey_script" ]; then
    # ファイルが存在すれば実行
    source "$setAPIKey_script"
else
    # ファイルが存在しない場合
    echo "setAPIKey.sh not found."
fi
