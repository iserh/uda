_default:
    just --list

install:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    poetry install

remote-install pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'install-session' 'just install'"

reinstall:
    eval "$(conda shell.bash hook)" && \
    conda create -n uda python=3.9 -y && \
    conda activate uda && \
    poetry install

remote-reinstall pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'install-session' 'just reinstall'"

bg +cmd:
    tmux new-session -d -s "bg-session" "just _bg {{cmd}}"
_bg +cmd:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    {{cmd}}

train pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; cd {{justfile_directory()}}; tmux new-session -d -s 'train-session' 'just _train'"
_train:
    eval "$(conda shell.bash hook)" && \
    conda activate uda && \
    python scripts/train_unet.py;

def_session := "train-session"
kill pool_nr session_name=def_session:
    ssh -t pool-u-042-{{pool_nr}} "tmux send-keys -t '{{session_name}}' 'C-c'"

ssh pool_nr:
    ssh pool-u-042-{{pool_nr}}

list-sessions pool_nr:
    ssh -t pool-u-042-{{pool_nr}} "tmux ls"

list-envs pool_nr:
    ssh -t pool-u-042-{{pool_nr}} ". ~/.bashrc; conda env list"
