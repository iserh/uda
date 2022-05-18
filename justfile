_default:
    just --list

install:
    conda create -n uda python=3.9 -y && \
    conda activate uda && \
    poetry install

download-cc359:
    tmux new-session -s "download-cc359" "just _dl_cc359"
_dl_cc359:
    mkdir '/tmp/data/CC359' --parents && poetry run gdown -O /tmp/data/CC359/ 1ODo9RQP3l14ZIr22yndrcapSS2LXkkiv

extract-cc359:
    python scripts/setup_cc359.py /tmp/data/CC359

setup-cc359:
    tmux new-session -s "setup-cc359" "just _dl_cc359 && just extract-cc359"

download-mms:
    tmux new-session -s "download-mms" "mkdir '/tmp/data/MMs' --parents && megatools dl --path /tmp/data/MMs  https://mega.nz/folder/FxAmhbRJ#Dwugf8isRSR9CCZ6Qnza4w"
