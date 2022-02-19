#! /usr/bin/env python3

import argparse
import os
import pathlib
import shutil
import subprocess
import tempfile
from collections import namedtuple
from dataclasses import dataclass

PackageMeta = namedtuple("PackageMeta", ["name", "tag", "url"])
PACKAGES_META = {
    "torch": PackageMeta("torch", "1.10.1", None),
    "cluster": PackageMeta("pytorch-cluster", "*", "https://github.com/rusty1s/pytorch_cluster.git"),
    "scatter": PackageMeta("pytorch-scatter", "2.0.8", "https://github.com/rusty1s/pytorch_scatter.git"),
    "sparse": PackageMeta("pytorch-sparse", "0.6.12", "https://github.com/rusty1s/pytorch_sparse.git"),
    "spline": PackageMeta("pytorch-spline-conv", "*", "https://github.com/rusty1s/pytorch_spline_conv.git"),
    "pyg": PackageMeta("pyg", "2.0.1", "https://github.com/pyg-team/pytorch_geometric.git"),
}


def run_command(command, run=True):
    print(" ".join(command))

    if not run:
        return

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for line in process.stdout:
            if process.poll() is not None:
                break
            if line:
                print(line.strip().decode("utf-8"))

        _, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8"))


def find_file(path, glob):
    results = sorted(pathlib.Path(path).glob(str(glob)))

    if len(results) > 1:
        raise RuntimeError(f"Multiple matches for glob '{str(glob)}' in path '{str(path)}'.\n{str(results)}")

    if len(results) < 1:
        raise RuntimeError(f"No matches for glob '{str(glob)}' in path '{str(path)}'.")

    return results[0]


def build_pytorch(args):
    print(f"Building PyTorch using the current feedstock with cuda_version={args.cuda}")

    tarfile = None
    if args.cuda == "none":
        run_command(
            [
                str((args.feedstock_dir / "build-locally.py").absolute()),
                str(
                    find_file(
                        args.feedstock_dir / ".ci_support", f"linux_64_*cuda_compiler_versionNone*python{args.python}*"
                    ).name
                )[:-5],
            ],
        )

        tarfile = find_file(
            args.feedstock_dir / "build_artifacts" / "linux-64",
            f"pytorch-{PACKAGES_META['torch'].tag}-cpu_py{args.python.replace('.','')}*_openmpi.*.bz2",
        )

    else:
        run_command(
            [
                str((args.feedstock_dir / "build-locally.py").absolute()),
                str(
                    find_file(
                        args.feedstock_dir / ".ci_support",
                        f"linux_64_*cuda_compiler_version{args.cuda}*python{args.python}*",
                    ).name
                )[:-5],
            ]
        )

        tarfile = find_file(
            args.feedstock_dir / "build_artifacts" / "linux-64",
            f"pytorch-{PACKAGES_META['torch'].tag}-cuda{args.cuda.replace('.', '')}"
            + f"py{args.python.replace('.','')}*_openmpi.*.bz2",
        )

    destination_path = pathlib.Path.home() / "conda-bld" / "linux-64/"
    shutil.copy(tarfile, destination_path)

    print("PyTorch built")
    return destination_path / tarfile.name


def get_line_after_match(file, match):
    lines = None
    with open(file, "r") as read_file:
        lines = read_file.readlines()

    for line in lines:
        index = line.find(match)
        if index != -1:
            return line[index + len(match) :].strip()
    raise RuntimeError(f"Match '{match}' not found in file '{file}'.")


class GitClone:
    """
    An instance of this class represents a remote repo cloned to the specified
    directory. It is recommended to use tmpfile with this class to both avoid
    namespace errors, and allow the OS to handle directory removal.
    """

    def __init__(self, directory, package):
        self.package = package
        self.directory = directory

        tag = PACKAGES_META[package].tag
        if tag == "*":
            tag = "master"
        run_command(["git", "clone", "-b", tag, PACKAGES_META[package].url, str(self.directory)])
        self.conda_dir = self.directory / "conda" / PACKAGES_META[package].name

    def get_version(self):
        return get_line_after_match(self.conda_dir / "meta.yaml", "version: ")

    def apply_patch(self, args):
        os.chdir(self.directory)
        for patch in (args.feedstock_dir / "pyg_support_patches" / self.package).iterdir():
            run_command(["git", "apply", str(args.feedstock_dir / "pyg_support_patches" / self.package / patch)])
        os.chdir(args.feedstock_dir)

    def build(self, args):
        os.chdir(self.conda_dir)
        if args.cuda == "none":
            run_command([str(self.conda_dir / "build_conda.sh"), args.python, PACKAGES_META["torch"].tag, "cpu"])
        else:
            run_command(
                [
                    str(self.conda_dir / "build_conda.sh"),
                    args.python,
                    PACKAGES_META["torch"].tag,
                    f"cu{args.cuda.replace('.', '')}",
                ]
            )
        os.chdir(args.feedstock_dir)


def build_package(package, args):
    print(f"Building package {package} with pytorch_version={PACKAGES_META['torch'].tag} and cuda_version={args.cuda}")
    version = None
    with tempfile.TemporaryDirectory() as tmpdir:
        feedstock = GitClone(pathlib.Path(tmpdir), package)
        version = feedstock.get_version()
        feedstock.apply_patch(args)
        feedstock.build(args)

    package_path = pathlib.Path.home().absolute() / "conda-bld" / "linux-64"

    if args.cuda == "none":
        tarfile = find_file(
            package_path,
            f"{PACKAGES_META[package].name}-{version}-py{args.python.replace('.', '')}"
            + f"_torch_{PACKAGES_META['torch'].tag}_*cpu_openmpi.tar.bz2",
        )
    else:
        tarfile = find_file(
            package_path,
            f"{PACKAGES_META[package].name}-{version}-py{args.python.replace('.', '')}"
            + f"_torch_{PACKAGES_META['torch'].tag}_*{args.cuda.replace('.', '')}_openmpi.tar.bz2",
        )
    print(f"Package {package} built")
    return tarfile


def build(args):
    package_locations = []
    threw_error = True
    try:
        if "torch" in args.package:
            args.package.remove("torch")
            package_locations.append(build_pytorch(args))

        should_build_pyg = False
        if "pyg" in args.package:
            args.package.remove("pyg")
            should_build_pyg = True

        for package in args.package:
            package_locations.append(build_package(package, args))

        if should_build_pyg:
            package_locations.append(build_package("pyg", args))

        threw_error = False
    finally:
        if threw_error:
            if len(package_locations) == 0:
                print("No packages built before exiting.")
            else:
                print("\nBuild incomplete.\nThe following packages were built before exiting.")
                for location in package_locations:
                    print(location)
                print()

    print("All packages built")
    with open(args.feedstock_dir / "package_locations.txt", "w") as location_file:
        for location in package_locations:
            location_file.write(f"{str(location)}\n")
    print(f"Package locations written to '{str(args.feedstock_dir)}/package_locations.txt'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds packages supporting PyG with OpenMPI")

    parser.add_argument(
        "-p",
        "--package",
        type=str,
        action="append",
        help="The package(s) to be built",
        choices=list(PACKAGES_META.keys()),
        default=None,
    )

    parser.add_argument(
        "--python",
        type=str,
        help="The version of python used when building packages.",
        choices=[f"3.{i}" for i in range(0, 12)],
        required=True,
    )

    parser.add_argument(
        "--cuda",
        type=str,
        help="The version of CUDA used when building packages.",
        choices=["none", "10.1", "10.2", "11.1", "11.2", "11.3"],
        required=True,
    )

    parser.add_argument(
        "--feedstock-dir",
        type=pathlib.Path,
        help="The directory of 'pytorch-cpu-feedstock'. Generally is the location of this file.",
        default=pathlib.Path(__file__).parent.absolute(),
    )

    args = parser.parse_args()
    if args.package is None:
        args.package = list(PACKAGES_META.keys())

    build(args)
