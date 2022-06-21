_default:
    just --list

remote cmd nr pool="042":
    ssh -t pool-u-{{pool}}-{{nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'remote-session' 'just {{cmd}}'"

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

train-unet nr pool="042":
    ssh -t pool-u-{{pool}}-{{nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'train-session' 'just _train-unet'"
_train-unet:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_unet.py;

train-vae nr pool="042":
    ssh -t pool-u-{{pool}}-{{nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'train-session' 'just _train-vae'"
_train-vae:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_vae.py;

kill nr pool="042" session_name="train-session":
    ssh -t pool-u-{{pool}}-{{nr}} "tmux send-keys -t '{{session_name}}' 'C-c'"

ssh nr pool="042":
    ssh pool-u-{{pool}}-{{nr}}

list-sessions nr pool="042":
    ssh -t pool-u-{{pool}}-{{nr}} "tmux ls"

list-envs nr pool="042":
    ssh -t pool-u-{{pool}}-{{nr}} ". ~/.bashrc; conda env list"
