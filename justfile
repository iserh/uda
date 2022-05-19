_default:
    just --list

install:
    eval "$(conda shell.bash hook)" && \
    conda create -n uda python=3.9 && \
    conda activate uda && \
    poetry install

download-cc359:
    tmux new-session -s "download-cc359" "just _dl_cc359"

setup-cc359:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/setup_cc359.py /tmp/data/CC359

all-cc359:
    tmux new-session -s "setup-cc359" "just _dl_cc359 && just extract-cc359"

download-mms:
    tmux new-session -s "download-mms" "just _dl_mms"


_dl_cc359:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    mkdir '/tmp/data/CC359' --parents && \
    gdown -O /tmp/data/CC359/ https://drive.google.com/file/d/1ODo9RQP3l14ZIr22yndrcapSS2LXkkiv/view?usp=sharing

_dl_mms:
    mkdir '/tmp/data/MMs' --parents && \
    megatools dl --path /tmp/data/MMs  https://mega.nz/folder/FxAmhbRJ#Dwugf8isRSR9CCZ6Qnza4w
