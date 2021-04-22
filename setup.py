"""setup script."""

from io import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def get_install_requires():
    """Read and parse requirements.txt."""
    with open("requirements.txt", "r") as f:
        lst = [
            x.strip()
            for x in f.read().split("\n")
            if not x.startswith("#") and not x.startswith("git+")
        ]
    return [x for x in lst if len(x) > 0]


def get_main_author():
    """Read authors from AUTHORS.md."""
    with open("AUTHORS.md", "r") as f:
        authors = f.readlines()
    main_author, main_email = authors[0].split(" : ")
    return main_author, main_email


def main():
    """Set up and run the setup function."""
    main_author, main_email = get_main_author()
    setup(
        name="cleaners",  # Required
        author=main_author,
        author_email=main_email,
        packages=find_packages(exclude=["tests", "notebooks"]),  # Required
        python_requires=">=3.6",
        install_requires=get_install_requires(),  # Optional
        include_package_data=True,
        package_data={"": ["resources/*"]},
    )


if __name__ == "__main__":
    main()
