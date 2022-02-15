#! /usr/bin/env python3

import argparse
import glob
import os
import pathlib
import signal
import subprocess
import tempfile
from collections import namedtuple
from dataclasses import dataclass


@dataclass
class PackagesMeta:
    Package = namedtuple("Package", ["key", "name", "tag", "url"])
    torch: Package = Package("torch", "torch", "1.10.1", None)
    cluster: Package = Package("cluster", "pytorch-cluster", "*", "https://github.com/rusty1s/pytorch_cluster.git")
    scatter: Package = Package("scatter", "pytorch-scatter", "2.0.8", "https://github.com/rusty1s/pytorch_scatter.git")
    sparse: Package = Package("sparse", "pytorch-sparse", "0.6.12", "https://github.com/rusty1s/pytorch_sparse.git")
    spline: Package = Package(
        "spline", "pytorch-spline-conv", "*", "https://github.com/rusty1s/pytorch_spline_conv.git"
    )
    pyg: Package = Package("pyg", "pyg", "2.0.1", "https://github.com/pyg-team/pytorch_geometric.git")


def run_command(command, run=True):
    command = "exec " + command
    print(command)

    if not run:
        return

    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, start_new_session=True
    ) as process:
        while True:
            output = process.stdout.readline()
            if process.poll() is not None:
                break
            if output:
                print(output.strip().decode("utf-8"))

        _, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8"))


def find_file(pathname):
    results = glob.glob(str(pathname))

    if len(results) > 1:
        raise RuntimeError("Multiple matches for glob pathname '" + str(pathname) + "'.\n" + str(results))

    if len(results) < 1:
        raise RuntimeError("No matches for glob pathname '" + str(pathname) + "'.")

    return results[0].split("/")[-1]


def build_pytorch(args):
    packages_meta = PackagesMeta()
    print("Building PyTorch using the current feedstock with cuda_version=" + args.cuda)

    args.python = "3.8"
    cuda = args.cuda
    if cuda == "cpu":
        cuda = "None"

    pattern = (
        args.feedstock_dir / ".ci_support" / ("linux_64_*cuda_compiler_version" + cuda + "*python" + args.python + "*")
    )
    run_command(str((args.feedstock_dir / "build-locally.py ").absolute()) + find_file(pattern)[:-5])

    if cuda == "None":
        cuda = "cpu_"
    else:
        cuda = "cuda" + cuda.replace(".", "")

    tarfile = find_file(
        args.feedstock_dir
        / "build_artifacts"
        / "linux-64"
        / ("pytorch-" + packages_meta.torch.tag + "-" + cuda + "py" + args.python.replace(".", "") + "*_openmpi.*.bz2")
    )
    destination_path = pathlib.Path.home() / "conda-bld" / "linux-64/"
    run_command(
        "cp " + str(args.feedstock_dir / "build_artifacts" / "linux-64" / tarfile) + " " + str(destination_path)
    )
    print("PyTorch built")
    return str(destination_path / tarfile)


def get_line_after_match(file, match):
    lines = None
    with open(file, "r") as read_file:
        lines = read_file.readlines()

    for line in lines:
        index = line.find(match)
        if index != -1:
            return line[index + len(match) :].strip()
    raise RuntimeError("Match '" + match + "' not found in file '" + file + "'.")


class GitClone:
    def __init__(self, directory, package):
        self.package = package
        self.directory = directory
        self.packages_meta = PackagesMeta()

        tag = getattr(self.packages_meta, package).tag
        if tag == "*":
            tag = "master"
        run_command("git clone -b " + tag + " " + getattr(self.packages_meta, package).url + " " + str(self.directory))
        self.conda_dir = self.directory / "conda" / getattr(self.packages_meta, package).name

    def get_version(self):
        return get_line_after_match(self.conda_dir / "meta.yaml", "version: ")

    def apply_patch(self, args):
        os.chdir(self.directory)
        run_command(
            "git apply " + str(args.feedstock_dir / "pyg_support_patches" / (self.package + ".meta.yaml.patch"))
        )
        run_command(
            "git apply " + str(args.feedstock_dir / "pyg_support_patches" / (self.package + ".build_conda.sh.patch"))
        )
        os.chdir(args.feedstock_dir)

    def build(self, args):
        cuda = args.cuda.replace(".", "")
        if cuda != "cpu":
            cuda = "cu" + cuda

        run_command(
            str(self.conda_dir / "build_conda.sh ") + args.python + " " + self.packages_meta.torch.tag + " " + cuda
        )


def build_package(package, args):
    packages_meta = PackagesMeta()
    print(
        "Building package "
        + package
        + " with pytorch_version="
        + packages_meta.torch.tag
        + " and cuda_version="
        + args.cuda
    )
    version = None
    with tempfile.TemporaryDirectory() as tmpdir:
        feedstock = GitClone(pathlib.Path(tmpdir), package)
        version = feedstock.get_version()
        feedstock.apply_patch(args)
        feedstock.build(args)

    package_path = pathlib.Path.home().absolute() / "conda-bld" / "linux-64"
    tarfile = find_file(
        package_path
        / (
            getattr(packages_meta, package).name
            + "-"
            + version
            + "-py"
            + args.python.replace(".", "")
            + "_torch_"
            + packages_meta.torch.tag
            + "_*"
            + args.cuda.replace(".", "")
            + "_openmpi.tar.bz2"
        )
    )
    print("Extension " + package + " built")
    return str(package_path / tarfile)


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
            location_file.write(location + "\n")
    print("Package locations written to '" + str(args.feedstock_dir) + "/package_locations.txt'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds packages supporting PyG with OpenMPI")

    parser.add_argument(
        "-p",
        "--package",
        type=str,
        action="append",
        help="The package(s) to be built",
        choices=list(PackagesMeta().__annotations__.keys()),
        default=None,
    )

    parser.add_argument(
        "--python", type=str, help="The version of python used when building packages.", choices=["3.8"], required=True
    )

    parser.add_argument(
        "--cuda",
        type=str,
        help="The version of CUDA used when building packages.",
        choices=["cpu", "10.2", "11.0", "11.1", "11.2"],
        required=True,
    )

    parser.add_argument(
        "--feedstock-dir",
        type=pathlib.Path,
        help="The directory of pytorch-cpu-feedstock. Generally is the location of this file.",
        default=pathlib.Path(__file__).parent.absolute(),
    )

    args = parser.parse_args()
    if args.package is None:
        args.package = list(PackagesMeta().__annotations__.keys())

    build(args)
