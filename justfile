_default:
    just --list

remote cmd server:
    ssh -t {{server}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'remote-session' 'just {{cmd}}'"

install:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    poetry install

reinstall:
    rm -rf /tmp/conda/envs/uda
    eval "$(conda shell.bash hook)" && \
    conda create -n uda python=3.9 -y && \
    conda activate uda && \
    poetry install

dl project run_id:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/download_run_config.py {{project}} {{run_id}}

bg +cmd:
    tmux new-session -d -s "bg-session" "just _bg {{cmd}}"
_bg +cmd:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    {{cmd}}

train-unet server:
    ssh -t {{server}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'train-session' 'just _train-unet'"
_train-unet:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_unet.py;

train-vae server:
    ssh -t {{server}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'train-session' 'just _train-vae'"
_train-vae:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_vae.py;

kill server session_name="train-session":
    ssh -t {{server}} "tmux send-keys -t '{{session_name}}' 'C-c'"

ssh server:
    ssh {{server}}

list-sessions server:
    ssh -t {{server}} "tmux ls"

list-envs server:
    ssh -t {{server}} ". ~/.bashrc; conda env list"
