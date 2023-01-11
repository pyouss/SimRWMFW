#!/usr/bin/python3

import os
import importlib

dependencies = ['scipy', 'configparser', 'numpy', 'pandas', 'matplotlib', 'unittest', 'mat73']

def check_and_install(dependencies):
    for package in dependencies:
        try:
            importlib.import_module(package)
            print(f'{package} is already installed.')
        except ImportError:
            install_package = input(f'{package} is not installed. Would you like to install it? (y/n)')
            if install_package.lower() == 'y':
                import subprocess
                subprocess.call(f'pip3 install {package}', shell=True)
            else:
                print(f'{package} is not installed.')
                exit()


if not os.path.exists("./regrets"):
    os.makedirs("regrets")



check_and_install(dependencies)

print("Set up is ready.")