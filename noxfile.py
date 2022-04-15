# coding: utf-8
"""Nox sessions."""
import configparser

import nox
from nox.sessions import Session

nox.options.sessions = (
    "format",
    "lint",
)  # set sessions for default call
nox.options.reuse_existing_virtualenvs = True  # -r option as default

locations = "src", "tests", "noxfile.py"
# location_with_files = "/src/**/*.py", "/tests/**/*.py", "noxfile.py"  # maybe obsolete
python_versions = ["3.9"]

# load line length from flake8 config
config = configparser.ConfigParser()
config.read(".flake8")
max_line_length = config["flake8"]["max-line-length"]


@nox.session(python=python_versions)
def lint(session: Session) -> None:
    """Lint using flake8.

    Args:
        session (Session): Nox session
    """
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-bugbear",
        "flake8-builtins",
        "flake8-docstrings",
        "flake8-isort",
        "flake8-use-fstring",
    )

    args = session.posargs or locations
    session.run("flake8", *args)


@nox.session(python=python_versions)
def format(session: Session) -> None:
    """Format code.

    Args:
        session (Session): Nox session
    """
    args = session.posargs or locations
    session.install("black", "isort", "docformatter", "reindent", "tomlkit")
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
    session.run("poetry", "run", "python", "pyproject_sort.py", external=True)
    session.run("black", "--line-length", f"{max_line_length}", *args)


@nox.session(python=python_versions)
def depsort(session: Session) -> None:
    """Sort the dependencies in pyproject.toml alphabetically.

    Args:
        session (Session): Nox session
    """
    session.install("tomlkit")
    session.run("poetry", "run", "python", "pyproject_sort.py", external=True)
