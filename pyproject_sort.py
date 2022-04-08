# coding: utf-8

from pathlib import Path

import tomlkit


def sort_table(table: tomlkit.items.Table) -> tomlkit.items.Table:
    """sort a tomlkit table.

    converts dicts to InlineTables
    """
    sorted_table = tomlkit.table()
    try:  # put python dependency in first positions
        python = table["python"]  # .pop() seems broken on tomlkit tables
        table.remove("python")
        sorted_table.add("python", python)
    except KeyError:
        pass
    for key, val in sorted(table.items(), key=lambda item: item[0].casefold()):
        if isinstance(val, dict):
            inline_table = tomlkit.inline_table()
            inline_table.update(val)
            val = inline_table
        sorted_table.add(key, val)
    return sorted_table


def read_toml(filename) -> tomlkit.toml_document.TOMLDocument:
    with open(filename, "r") as infile:
        return tomlkit.parse(infile.read())


def dump_toml(filename: Path, toml_doc: tomlkit.toml_document.TOMLDocument) -> None:
    with open(filename, "w") as f:
        f.write(tomlkit.dumps(toml_doc))


def sort_file(filename: Path):
    """Sort dependency sections in a file in place."""
    doc = read_toml(filename)
    dependencies = doc["tool"]["poetry"]["dependencies"]
    doc["tool"]["poetry"]["dependencies"] = sort_table(dependencies)
    dev_dependencies = doc["tool"]["poetry"]["dev-dependencies"]
    doc["tool"]["poetry"]["dev-dependencies"] = sort_table(dev_dependencies)
    dump_toml(filename, doc)


def sort_pyproject():
    sort_file(Path("pyproject.toml"))


if __name__ == "__main__":
    sort_pyproject()
    # sort_file(Path("pyproject_copy.toml"))
