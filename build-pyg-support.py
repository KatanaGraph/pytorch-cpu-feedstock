#! /usr/bin/env python3

import subprocess
from threading import Timer
import os
import signal
import tempfile
import argparse
import re
import warnings

PACKAGES = {
    'torch': {
        'tag': '1.10.1'
    },
    'cluster': {
        'tag': '*',
        'url': 'https://github.com/rusty1s/pytorch_cluster.git',
        'conda_path': '/conda/pytorch-cluster/'
    },
    'scatter': {
        'tag': '2.0.8',
        'url': 'https://github.com/rusty1s/pytorch_scatter.git',
        'conda_path': '/conda/pytorch-scatter/'
    },
    'sparse': {
        'tag': '0.6.12',
        'url': 'https://github.com/rusty1s/pytorch_sparse.git',
        'conda_path': '/conda/pytorch-sparse/',
        'metis': False
    },
    'spline-conv': {
        'tag': '*',
        'url': 'https://github.com/rusty1s/pytorch_spline_conv.git',
        'conda_path': '/conda/pytorch-spline-conv/'
    },
    'pyg': {
        'tag': '2.0.1',
        'url': 'https://github.com/pyg-team/pytorch_geometric.git',
        'conda_path': '/conda/pyg/'
    }
}

def run_command(command, timeout=None, run=True):
    command = 'exec '+command
    print(command)

    if not run:
        return

    def kill_proc(process, cancelled):
        cancelled[0] = True
        os.killpg(os.getpgid(process.pid), signal.SIGINT)

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, start_new_session=True)
    timer = None
    cancelled = [False]
    if timeout is not None:
        timer = Timer(timeout, kill_proc, [process, cancelled])
        timer.start()
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode("utf-8"))

    if timer is not None:
        timer.cancel()
    _, stderr = process.communicate()

    if cancelled[0]:
        warnings.warn("Command timed out: "+command, RuntimeWarning)
    elif process.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8"))

def find_file(root, pattern):
    result = None
    for name in os.listdir(root):
        if re.search(pattern, name):
            if result is not None:
                raise RuntimeError(
                    "Regex found multiple matches for pattern '"+pattern+"' in '"+root+"'."
                )
            result = name

    if result is None:
        raise RuntimeError(
            "Regex found no matches for pattern '"+pattern+"' in '"+root+"'."
        )

    return result

def build_pytorch(args):
    print('Building PyTorch using the current feedstock with cuda_version='+args.cuda)

    args.python = '3.8'
    cuda = args.cuda
    if cuda == 'cpu':
        cuda = 'None'

    config_dir = args.feedstock_dir+'.ci_support/'
    pattern = '^linux_64_.*cuda_compiler_version'+cuda+'.*python'+args.python

    run_command(args.feedstock_dir+'/build-locally.py '+find_file(config_dir, pattern)[:-5])

    if cuda == 'None':
        cuda = 'cpu_'
    else:
        cuda = 'cuda'+cuda.replace('.', '')

    tarfile = find_file(args.feedstock_dir+"/build_artifacts/linux-64/", 'pytorch-'+PACKAGES['torch']['tag']+'-'+cuda+'py'+args.python.replace('.', '')+'.*_openmpi.*.bz2')
    destination_path = os.path.expanduser('~')+"/conda-bld/linux-64/"
    run_command('cp '+args.feedstock_dir+"/build_artifacts/linux-64/"+tarfile+' '+destination_path)
    print("PyTorch built")
    return(destination_path+tarfile)

class GitClone:
    # Downloads, modifies, and builds repos
    def __init__(self, directory, package, output_folder):
        self.package = package
        self.directory = directory
        run_command('rm -rf '+self.directory)
        self.output_folder = output_folder

        tag = PACKAGES[package]['tag']
        if tag == '*':
            tag = 'master'
        run_command('git clone -b '+tag+' '+PACKAGES[package]['url']+' '+self.directory)
        self.conda_dir = self.directory+PACKAGES[package]['conda_path']

    def get_version(self):
        meta = open(self.conda_dir+'meta.yaml', 'r')
        lines = meta.readlines()

        for line in lines:
            index = line.find('version: ')
            if index != -1:
                return line[index+len('version: '):].strip()
        raise RuntimeError("Version of package '"+self.package+"' not found.")

    def sed_extension(self):
        # remove defaults channel
        run_command("sed -i 's: -c defaults::g' "+self.conda_dir+"build_conda.sh")
        # add katanagraph channel
        run_command("sed -i 's:^conda build . :conda build "+self.conda_dir+" --override-channels -c file\://"+os.path.expanduser('~')+"/conda-bld/linux-64/"+" :g' "+self.conda_dir+"build_conda.sh")
        # set the output folder
        # tests do not pass when changing the output folder directory
        #run_command("sed -i 's:--output-folder.*$:--croot "+self.output_folder+" :g' "+self.conda_dir+"build_conda.sh")
        # require our version of pytorch
        run_command("sed -i 's:pytorch==${TORCH_VERSION%.\*}.\*:pytorch==${TORCH_VERSION} \*_openmpi:g' "+self.conda_dir+"build_conda.sh")
        # append build string with '_openmpi'
        run_command("sed -i '/string:/ s/$/_openmpi/g' "+self.conda_dir+"meta.yaml")
        # pytorch-sparse can use metis to install all functionality. Requires sudo
        if 'metis' in PACKAGES[self.package].keys() and PACKAGES[self.package]['metis']:
            run_command(self.directory+'.github/workflows/metis.sh')
        else:
            run_command("sed -i 's/- WITH_METIS=1/- WITH_METIS=0/g' "+self.conda_dir+"meta.yaml")

        # use mamba instead of conda
        run_command("sed -i 's:conda build:mamba build:g' "+self.conda_dir+"build_conda.sh")

    def sed_pyg(self):
        self.sed_extension()
        # Require our build of extensions
        for package in ['cluster', 'scatter', 'sparse', 'spline-conv']:
            run_command("sed -i 's:- pytorch-"+package+":- pytorch-"+package+" \* \*_openmpi:g' "+self.conda_dir+"meta.yaml")

    def build(self, args):
        cuda = args.cuda.replace('.', '')
        if cuda != 'cpu':
            cuda = 'cu'+cuda
        run_command(self.conda_dir+'build_conda.sh '+args.python+' '+PACKAGES['torch']['tag']+' '+cuda)

def build_extension(package, args):
    # Relevent to non PyTorch or PyG packages
    print("Building extension "+package+" with pytorch_version="+PACKAGES['torch']['tag']+" and cuda_version="+args.cuda)
    version = None
    with tempfile.TemporaryDirectory() as tmpdir:
        feedstock = GitClone(tmpdir, package, args.feedstock_dir+"/build_artifacts/linux-64/")
        version = feedstock.get_version()
        feedstock.sed_extension()
        feedstock.build(args)

    package_path = os.path.expanduser('~')+"/conda-bld/linux-64/"
    tarfile = find_file(package_path, 'pytorch-'+package+'-'+version+'-py'+args.python.replace('.', '')+'_torch_'+PACKAGES['torch']['tag']+'_.*'+args.cuda.replace('.', '')+'_openmpi.tar.bz2')
    print("Extension "+package+" built")
    return(package_path+tarfile)

def build_pyg(args):
    print("Building PyG with pytorch_version="+PACKAGES['torch']['tag']+" and cuda_version="+args.cuda)
    version = None
    with tempfile.TemporaryDirectory() as tmpdir:
        feedstock = GitClone(tmpdir, 'pyg', args.feedstock_dir+"/build_artifacts/linux-64/")
        version = feedstock.get_version()
        feedstock.sed_pyg()
        feedstock.build(args)

    package_path = os.path.expanduser('~')+"/conda-bld/linux-64/"
    tarfile = find_file(package_path, 'pyg-'+version+'-py'+args.python.replace('.', '')+'_torch_'+PACKAGES['torch']['tag']+'_.*'+args.cuda.replace('.', '')+'_openmpi.tar.bz2')
    print("PyG built")
    return(package_path+tarfile)

def build(args):
    package_locations = []
    threw_error = True
    try:
        if 'torch' in args.package:
            args.package.remove('torch')
            package_locations.append(build_pytorch(args))

        should_build_pyg = False
        if 'pyg' in args.package:
            args.package.remove('pyg')
            should_build_pyg = True

        for extension in args.package:
            package_locations.append(build_extension(extension, args))

        if should_build_pyg:
            package_locations.append(build_pyg(args))

        threw_error=False
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
    location_file = open(args.feedstock_dir+'/package_locations.txt', 'w')
    for location in package_locations:
        location_file.write(location+'\n')
    location_file.close()
    print("Package locations written to '"+args.feedstock_dir+"/package_locations.txt'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Builds packages supporting PyG with OpenMPI')

    parser.add_argument(
        '-p',
        '--package',
        type = str,
        action = 'append',
        help = 'The package(s) to be built',
        choices = PACKAGES.keys(),
        default = None
    )

    parser.add_argument(
        '--python',
        type = str,
        help = 'The version of python used when building packages.',
        choices = ['3.8'],
        required = True
    )

    parser.add_argument(
        '--cuda',
        type = str,
        help = 'The version of CUDA used when building packages.',
        choices = ['cpu', '10.2', '11.0', '11.1', '11.2'],
        required = True
    )

    parser.add_argument(
        '--feedstock-dir',
        type = str,
        help = 'The directory of pytorch-cpu-feedstock. Generally is the location of this file.',
        default = os.path.dirname(os.path.abspath(__file__))
    )

    args = parser.parse_args()
    args.feedstock_dir += '/'
    if args.package is None:
        args.package = list(PACKAGES.keys())

    build(args)
