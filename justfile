_default:
    just --list

install:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    poetry install

remote-install pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'install-session' 'just install'"

reinstall:
    rm -rf /tmp/conda/envs/uda
    eval "$(conda shell.bash hook)" && \
    conda create -n uda python=3.9 -y && \
    conda activate uda && \
    poetry install

remote-reinstall pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'install-session' 'just reinstall'"

download project run_id:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/download_run_config.py {{project}} {{run_id}}

bg +cmd:
    tmux new-session -d -s "bg-session" "just _bg {{cmd}}"
_bg +cmd:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    {{cmd}}

train-u pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'train-session' 'just _train-u'"
_train-u:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_unet.py;

train-v pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'train-session' 'just _train-v'"
_train-v:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_vae.py;

def_session := "train-session"
kill pool_nr session_name=def_session:
    ssh -t pool-u-042-{{pool_nr}} "tmux send-keys -t '{{session_name}}' 'C-c'"

ssh pool_nr:
    ssh pool-u-042-{{pool_nr}}

list-sessions pool_nr:
    ssh -t pool-u-042-{{pool_nr}} "tmux ls"

list-envs pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; conda env list"
