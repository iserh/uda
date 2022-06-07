_default:
    just --list

install:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    poetry install

reinstall:
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

train pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'bg-session' 'just _train'"
_train:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_unet.py;

kill pool_nr:
    ssh -t pool-u-042-{{pool_nr}} "tmux send-keys -t 'bg-session' 'C-c'"

ssh pool_nr:
    ssh pool-u-042-{{pool_nr}}
