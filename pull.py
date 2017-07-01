from clint.textui import colored, puts, indent

import os
import config
import credentials
import subprocess
from config import *
import pandas as pd
import shutil
from tqdm import tqdm

def command(command, path, log=True):
    if log:
        subprocess.call([command], shell=True, cwd=path)
    else:
        subprocess.call([command], shell=True, cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

directory_structure = [
    config.tmp_dir,
    config.images_path,
    config.training_data_dir,
    config.original_jpg_image_path,
    config.original_jpg_train_image_path,
    config.original_jpg_test_image_path,
    config.original_tif_image_path,
    config.original_tif_train_image_path,
    config.original_tif_test_image_path,
    config.broken_tif_test_image_path
]

def validate_directory_structure(directories):
    """
    Recursively attempt to create directories
    """
    try:
        head, *tail = directories

        if not head:
            puts(colored.cyan('directories ok'))
            return

        if not os.path.isdir(head):
            puts(colored.red('creating {}'.format(head)))
            command('mkdir {}'.format(head), config.run_dir)

        validate_directory_structure(tail)

    except ValueError: pass

def pull_unzip(file, target, test_file, verbose):

    if not os.path.isfile(target + '/' + test_file):
        puts(colored.red('downloading {}'.format(file)))
        tar_name = file.replace(".7z", "")
        dir_name = config.tmp_dir + tar_name.replace(".tar", "") + '/*'
        download_kaggle_file(file, config.tmp_dir)
        unzip_7z(file, config.tmp_dir)
        unzip_tar(tar_name, config.tmp_dir)
        move_imgs(dir_name, target, config.tmp_dir)

        if not os.path.isfile(target + '/' + test_file):
            puts(colored.red('fail - {}'.format(file)))
            return

    if verbose: puts(colored.cyan('ok - {}'.format(file)))


def validate_images(verbose=False):

    pull_unzip('train-jpg.tar.7z', config.original_jpg_train_image_path, 'train_0.jpg', verbose)
    pull_unzip('train-tif-v2.tar.7z', config.original_tif_train_image_path, 'train_0.tif', verbose)
    pull_unzip('test-jpg.tar.7z', config.original_jpg_test_image_path, 'test_0.jpg', verbose)
    pull_unzip('test-jpg-additional.tar.7z', config.original_jpg_test_image_path, 'file_0.jpg', verbose)
    pull_unzip('test-tif-v2.tar.7z', config.broken_tif_test_image_path, 'test_0.tif', verbose)

def fix_test_tifs():

    if not os.path.isfile(config.original_tif_test_image_path + '/file_0.tif'):

        df = pd.read_csv(tmp_dir + 'test_v2_file_mapping.csv')
        for index, row in tqdm(df.iterrows()):
            old = os.path.join(broken_tif_test_image_path, row['old'])
            new = os.path.join(original_tif_test_image_path, row['new'])
            shutil.copy(old, new)

    if not os.path.isfile(config.original_tif_test_image_path + '/test_0.tif'):
        tifs = os.listdir(broken_tif_test_image_path)

        for tif in tqdm(tifs):
            if 'test' not in tif: continue
            old = os.path.join(broken_tif_test_image_path, tif)
            new = os.path.join(original_tif_test_image_path, tif)
            shutil.copy(old, new)

def validate_train_csv(tries=1, verbose=False):

    if tries == 0:
        if verbose: puts(colored.green('failed to create training csv'))

    ## CHECK IF FILE EXISTS
    if not os.path.isfile(config.train_csv_path):
        puts(colored.red('downloading training csv'))
        download_kaggle_file('train_v2.csv.zip', config.tmp_dir)
        remove_macosx('train_v2.csv.zip', config.tmp_dir)
        unzip('train_v2.csv.zip', config.tmp_dir)
        move_from_tmp('train_v2.csv', config.tmp_dir, config.train_csv_path)
        return validate_train_csv(tries - 1)

    if verbose: puts(colored.cyan('ok - training csv'))


def validate_mapping_csv(tries=1, verbose=True):

    if tries == 0:
        if verbose: puts(colored.green('failed to create mapping csv'))

    ## CHECK IF FILE EXISTS
    if not os.path.isfile(config.tmp_dir + 'test_v2_file_mapping.csv'):
        puts(colored.red('downloading test vs mapping csv'))
        download_kaggle_file('test_v2_file_mapping.csv.zip', config.tmp_dir)
        remove_macosx('test_v2_file_mapping.csv.zip', config.tmp_dir)
        unzip('test_v2_file_mapping.csv.zip', config.tmp_dir)
        move_from_tmp('test_v2_file_mapping.csv', config.tmp_dir, config.tmp_dir + 'test_v2_file_mapping.csv')
        return validate_train_csv(tries=(tries - 1))

    if verbose: puts(colored.cyan('ok - mapping csv'))


def download_kaggle_file(file, path):

    if not os.path.isfile(path + file):

        command('kg download -u {} -p {} -c {} -f {}'.format(
            credentials.login['username'],
            credentials.login['password'],
            credentials.login['competition'],
            file
        ), path)
    else:
        puts(colored.cyan('defaulting to local {}'.format(file)))

def unzip_7z(file, path):
    print('7z x {} -aos'.format(file))
    expected_name = file.replace(".7z", "")
    if not os.path.isfile(path + expected_name):
        puts(colored.cyan('* unzipping 7z'))
        command('7z x {} -aos'.format(file), path)
    else:
        puts(colored.cyan('defaulting to local {}'.format(expected_name)))

def unzip(file, path):
    print('unzip {}'.format(file))
    expected_name = file.replace(".zip", "")
    if not os.path.isfile(path + expected_name):
        puts(colored.cyan('* unzipping'))
        command('unzip {}'.format(file), path)
    else:
        puts(colored.cyan('defaulting to local {}'.format(expected_name)))

def unzip_tar(file, path):
    print('tar -zxvf {}'.format(file))
    expected_name = file.replace(".tar", "")
    if not os.path.isfile(path + expected_name):
        puts(colored.cyan('* unzipping Tar'))
        command('tar -zxvf {}'.format(file), path)
    else:
        puts(colored.cyan('defaulting to local {}'.format(expected_name)))

def unzip(file, path):
    print('unzip {}'.format(file))
    expected_name = file.replace(".zip", "")
    if not os.path.isfile(path + expected_name):
        puts(colored.cyan('* unzipping zip'))
        command('unzip {}'.format(file), path)
    else:
        puts(colored.cyan('defaulting to local {}'.format(expected_name)))

def remove_macosx(file, path):
    command('zip -d {} \"__MACOSX*\"'.format(file), path, log=False)

def move_from_tmp(file, path, new):
    puts(colored.cyan('* moving files'))
    print('mv {} {}'.format(file, new))
    command('mv {} {}'.format(file, new), path, log=False)

def move_imgs(source, target, path):
    puts(colored.cyan('* moving files'))
    command('for file in {}; do mv -- "$file" {} ; done'.format(source, target), path, log=False)

def clean_tmp():
    command('rm -r tmp/*', config.run_dir)


def initial_check():
    validate_directory_structure(directory_structure)


def pull():

    with indent(4):

        puts(colored.magenta('\npull\n'))

        validate_directory_structure(directory_structure)

        validate_train_csv(1)
        validate_mapping_csv(1)
        validate_images()

        puts('\n')
