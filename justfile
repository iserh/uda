_default:
    just --list

install:
    eval "$(conda shell.bash hook)" && \
    conda create -n uda python=3.9 && \
    conda activate uda && \
    poetry install


setup-run-config:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/setup_run_config.py


bg +cmd:
    tmux new-session -s "bg-session" "just _bg {{cmd}}"
_bg +cmd:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    {{cmd}}


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
