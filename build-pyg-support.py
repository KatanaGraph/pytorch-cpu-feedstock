#! /usr/bin/env python3

import subprocess
import os
import yaml
import argparse
import re

PACKAGES = {
    'torch': {
        'tag': '1.9'
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
        'conda_path': '/conda/pytorch-sparse/'
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

def run_command(command, stdout=subprocess.PIPE, shell=False, run=True):
    if shell:
        print('command: '+command)
    else:
        print('command: '+' '.join(command))

    if not run:
        return

    process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.PIPE, shell=shell)
    stdout, stderr = process.communicate()
    if stdout != None and len(stdout) > 0:
        print(stdout.decode("utf-8"))

    if process.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8"))

def upload_package(path):
    run_command('anaconda upload --user KatanaGraph -l test '+path)

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
            "Regex found no configs for pattern '"+pattern+"' in '"+root+"'."
        )

    return result

def build_pytorch(args):
    print('Building PyTorch using the current feedstock with cuda_version='+args.cuda)

    args.python = '3.8'
    if args.cuda == 'cpu':
        args.cuda = 'None'

    config_dir = args.feedstock_dir+'.ci_support/'
    pattern = '^linux_64_cuda_compiler_version'+args.cuda+'.*python'+args.python

    run_command(args.feedstock_dir+'/build-locally.py '+find_file(config_dir, pattern)[:-5], shell=True, run=False)
    print("PyTorch built")

class GitClone:
    # Downloads, modifies, and deletes repos
    def __init__(self, root, package, output_folder, clean=False):
        self.output_folder = output_folder
        url = PACKAGES[package]['url']
        self.directory = root+url.split('/')[-1][:-4]+'_tmp'
        if clean:
            run_command(['rm', '-rf', self.directory])

        tag = PACKAGES[package]['tag']
        if tag == '*':
            tag = 'master'
        run_command(['git', 'clone', '-b', tag, url, self.directory])
        self.conda_dir = self.directory+PACKAGES[package]['conda_path']

    def sed_extension(self):
        # add katanagraph channel
        run_command("sed -i 's:^conda build . :conda build . -c katanagraph :g' "+self.conda_dir+"build_conda.sh", shell=True)
        # set the output folder
        run_command("sed -i 's:--output-folder.*$:--output-folder "+self.output_folder+" :g' "+self.conda_dir+"build_conda.sh", shell=True)
        # require our version of pytorch
        run_command("sed -i 's:pytorch==${TORCH_VERSION%.\*}.\*:pytorch==${TORCH_VERSION%.\*}.\* \*_openmpi:g' "+self.conda_dir+"build_conda.sh", shell=True)
        # append build string with '_openmpi'
        run_command("sed -i '/string:/ s/$/_openmpi/g' "+self.conda_dir+"meta.yaml", shell=True)

    def sed_pyg(self):
        self.sed_extension()
        # Require our version of extensions
        for package in ['cluster', 'scatter', 'sparse', 'spline-conv']:
            run_command("sed -i 's:- pytorch-"+package+":- pytorch-"+package+" \* \*_openmpi:g' "+self.conda_dir+"meta.yaml", shell=True)

    def build(self, args):
        run_command([self.conda_dir+'build_conda.sh', args.python, PACKAGES['torch']['tag'], args.cuda], run=False)

    def __del__(self):
        run_command(['rm', '-rf', self.directory], run=False)

def build_extension(package, args):
    # Relevent to non PyTorch or PyG packages
    print("Building extension "+package+" with pytorch_version="+PACKAGES['torch']['tag']+" and cuda_version="+args.cuda)
    feedstock = GitClone(args.feedstock_dir, package, args.feedstock_dir+"/build_artifacts/linux-64/", args.clean)
    feedstock.sed_extension()
    try:
        feedstock.build(args)
    finally:
        del feedstock

    tarfile = find_file(args.feedstock_dir+"/build_artifacts/linux-64/", 'pytorch-'+package+'.*py'+args.python.replace('.', '')+'_torch_'+PACKAGES['torch']['tag']'_'+args.cuda+'_openmpi.*.bz2')
    upload_package(args.feedstock_dir+"/build_artifacts/linux-64/"+tarfile)
    print("Extension "+package+" built")

def build_pyg(args):
    print("Building PyG with pytorch_version="+PACKAGES['torch']['tag']+" and cuda_version="+args.cuda)
    feedstock = GitClone(args.feedstock_dir, 'pyg', args.feedstock_dir+"/build_artifacts/linux-64/", args.clean)
    feedstock.sed_pyg()
    try:
        feedstock.build(args)
    finally:
        del feedstock

    tarfile = find_file(args.feedstock_dir+"/build_artifacts/linux-64/", 'pytorch-'+package+'.*py'+args.python.replace('.', '')+'_torch_'+PACKAGES['torch']['tag']'_'+args.cuda+'_openmpi.*.bz2')
    upload_package(args.feedstock_dir+"/build_artifacts/linux-64/"+tarfile)
    print("PyG built")

def build(config):
    if 'torch' in config.package:
        config.package.remove('torch')
        build_pytorch(args)

    should_build_pyg = False
    if 'pyg' in config.package:
        config.package.remove('pyg')
        should_build_pyg = True

    for extension in config.package:
        build_extension(extension, args)

    if should_build_pyg:
        build_pyg(args)

    print("All packages built")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Builds packages supporting PyG with OpenMPI')

    parser.add_argument(
        '-p',
        '--package',
        type = str,
        action = 'append',
        help = 'The package(s) to be built',
        choices = PACKAGES.keys(),
        required = True
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

    parser.add_argument(
        '--clean',
        help = 'If included, removes directories in the namespace of downloaded git repos.',
        action = 'store_true',
        default = False
    )

    args = parser.parse_args()
    args.feedstock_dir += '/'
    build(args)
