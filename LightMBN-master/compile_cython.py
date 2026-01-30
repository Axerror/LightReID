#!/usr/bin/env python
"""
Cython compilation script for rank_cylib module.
Run this script whenever you modify rank_cy.pyx

Usage:
    python compile_cython.py          # Normal compilation
    python compile_cython.py --clean  # Clean and rebuild
"""

import os
import sys
import shutil
import subprocess
import argparse


def clean_build_artifacts():
    """Remove old build artifacts"""
    rank_cylib_dir = os.path.join(os.path.dirname(__file__), 'utils', 'rank_cylib')
    
    # Directories and files to clean
    artifacts = [
        os.path.join(rank_cylib_dir, 'build'),
        os.path.join(rank_cylib_dir, 'rank_cy.c'),
        os.path.join(rank_cylib_dir, '__pycache__'),
    ]
    
    # Remove .pyd files (compiled for current Python version)
    pyd_pattern = os.path.join(rank_cylib_dir, 'rank_cy.*.pyd')
    import glob
    artifacts.extend(glob.glob(pyd_pattern))
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
                print(f'[INFO] Removed directory: {artifact}')
            else:
                os.remove(artifact)
                print(f'[INFO] Removed file: {artifact}')
    
    print('[INFO] Cleanup completed')


def compile_cython():
    """Compile Cython extension"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print('[INFO] Starting Cython compilation...')
    print(f'[INFO] Working directory: {project_root}')
    
    result = subprocess.run(
        [sys.executable, 'utils/rank_cylib/setup.py', 'build_ext', '--inplace'],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print('[ERROR] Cython compilation failed!')
        sys.exit(1)
    
    print('[SUCCESS] Cython module compiled successfully!')
    print('[INFO] You can now run: python main.py --config lmbn_config.yaml')


def main():
    parser = argparse.ArgumentParser(description='Compile Cython extension for rank_cylib')
    parser.add_argument('--clean', action='store_true', help='Clean build artifacts before compiling')
    args = parser.parse_args()
    
    if args.clean:
        clean_build_artifacts()
    
    compile_cython()


if __name__ == '__main__':
    main()
