_default:
    just --list

install:
    eval "$(conda shell.bash hook)" && \
    conda create -n uda python=3.9 && \
    conda activate uda && \
    poetry install


# --------------------------------------------------
# ----- mlflow -----
# --------------------------------------------------
backend_store_uri := "${MLFLOW_BACKEND_STORE_URI:-$HOME/data/mlruns}"
default_artifact_root := "${MLFLOW_ARTIFACT_STORE_URI:-/tmp/data/artifacts}"
ui_port := '${MLFLOW_UI_PORT:-5001}'

server backend_path=backend_store_uri artifact_path=default_artifact_root:
    tmux new-session -s "mlflow-server" "just _server {{backend_path}} {{artifact_path}}"
_server backend_path=backend_store_uri artifact_path=default_artifact_root:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    mlflow server --backend-store-uri "file://{{backend_path}}" --default-artifact-root "file://{{artifact_path}}"

ui port=ui_port backend_path=backend_store_uri artifact_path=default_artifact_root:
    tmux new-session -s "mlflow-ui" "just _ui {{port}} {{backend_path}} {{artifact_path}}"
_ui port=ui_port backend_path=backend_store_uri artifact_path=default_artifact_root:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    mlflow ui -p {{port}} --backend-store-uri "file://{{backend_path}}" --default-artifact-root "file://{{artifact_path}}"

gc backend_path=backend_store_uri:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/mlflow_gc.py --backend-store-uri "file://{{backend_path}}"


# --------------------------------------------------
# ----- Datasets -----
# --------------------------------------------------
cc359_path := '/tmp/data/CC359'
mms_path := '/tmp/data/MMs'

download-cc359 path=cc359_path:
    tmux new-session -s "download-cc359" "just _dl_cc359 {{path}}"
_dl_cc359 path=cc359_path:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    mkdir {{path}} --parents && \
    gdown -O {{path}} https://drive.google.com/file/d/1ODo9RQP3l14ZIr22yndrcapSS2LXkkiv/view?usp=sharing

setup-cc359 path=cc359_path:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/setup_cc359.py {{path}}

all-cc359 path=cc359_path:
    tmux new-session -s "all-cc359" "just _dl_cc359 {{path}} && just setup-cc359 {{path}}"


download-mms path=mms_path:
    tmux new-session -s "download-mms" "just _dl_mms {{path}}"
_dl_mms path=mms_path:
    mkdir {{path}} --parents && \
    megatools dl --path {{path}}  https://mega.nz/folder/FxAmhbRJ#Dwugf8isRSR9CCZ6Qnza4w

