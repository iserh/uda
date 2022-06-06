_default:
    just --list

install:
    eval "$(conda shell.bash hook)" && \
    conda create -n uda python=3.9 && \
    conda activate uda && \
    poetry install

bg +cmd:
    tmux new-session -d -s "bg-session" "just _bg {{cmd}}"
_bg +cmd:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    {{cmd}}
