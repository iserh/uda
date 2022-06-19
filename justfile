_default:
    just --list

remote nr cmd:
    ssh -t pool-u-042-{{nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'remote-session' 'just {{cmd}}'"

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

train-unet nr:
    ssh -t pool-u-042-{{nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'train-session' 'just _train-unet'"
_train-unet:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_unet.py;

train-vae nr:
    ssh -t pool-u-042-{{nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'train-session' 'just _train-vae'"
_train-vae:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_vae.py;

def_session := "train-session"
kill nr session_name=def_session:
    ssh -t pool-u-042-{{nr}} "tmux send-keys -t '{{session_name}}' 'C-c'"

ssh nr:
    ssh pool-u-042-{{nr}}

list-sessions nr:
    ssh -t pool-u-042-{{nr}} "tmux ls"

list-envs nr:
    ssh -t pool-u-042-{{nr}} ". ~/.bashrc; conda env list"
