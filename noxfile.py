# coding: utf-8
"""Nox sessions."""
import configparser

import nox
from nox.sessions import Session

nox.options.sessions = (
    "format",
    "lint",
)  # set sessions for default call

locations = "uda", "uda_wandb", "tests", "scripts", "noxfile.py"

# load line length from flake8 config
config = configparser.ConfigParser()
config.read(".flake8")
max_line_length = config["flake8"]["max-line-length"]


@nox.session(python=False)
def lint(session: Session) -> None:
    args = session.posargs or locations

    session.run("flake8", *args)


@nox.session(python=False)
def format(session: Session) -> None:
    args = session.posargs or locations

    session.run("isort", "--atomic", *args)
    session.run(
        "docformatter",
        "--wrap-summaries",
        f"{max_line_length}",
        "--wrap-descriptions",
        f"{max_line_length}",
        "--in-place",
        "--recursive",
        *args,
    )
    session.run("python", "-m", "reindent", "-r", "-n", *args)
    session.run("black", "--line-length", f"{max_line_length}", *args)
