

## TODO Check for existance of full scale images
## TODO Report diagnostic
## TODO If full scale jpegs don't exist redownload
## TODO Check fo existance of tifs (For NIR)
## TODO If tifs don't exist redownload

from clint.textui import colored, puts, indent

import os
import config
import credentials
import subprocess

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
    config.original_tif_test_image_path
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


def validate_original_jpg_train_images(tries):
    if tries == 0:
        puts(colored.cyan('failed to validate training images'))
        return False

    if not os.path.isfile(config.original_jpg_train_image_path + '/train_0.jpg'):
        puts(colored.red('downloading original jpgs'))
        download_kaggle_file('train-jpg.tar.7z', config.tmp_dir)
        unzip_7z('train-jpg.tar.7z', config.tmp_dir)
        unzip_tar('train-jpg.tar', config.tmp_dir)
        move_imgs('./train-jpg/*', config.original_jpg_train_image_path, config.tmp_dir)

    puts(colored.cyan('full size jpgs ok'))
    return True

def validate_original_tif_train_images(tries):
    if tries == 0:
        puts(colored.cyan('failed to validate tif training images'))
        return False

    if not os.path.isfile(config.original_tif_train_image_path + '/train_0.tif'):
        puts(colored.red('downloading original train tifs'))
        download_kaggle_file('train-tif-v2.tar.7z', config.tmp_dir)
        unzip_7z('train-tif-v2.tar.7z', config.tmp_dir)
        unzip_tar('train-tif-v2.tar', config.tmp_dir)
        move_imgs('./train-tif-v2/*', config.original_tif_train_image_path, config.tmp_dir)

    puts(colored.cyan('full size tifs ok'))
    return True


def validate_original_jpg_test_images(tries):
    if tries == 0:
        puts(colored.cyan('failed to validate training images'))
        return False

    if not os.path.isfile(config.original_jpg_test_image_path + '/test_0.jpg'):
        puts(colored.red('downloading original test jpgs'))
        download_kaggle_file('test-jpg.tar.7z', config.tmp_dir)
        unzip_7z('test-jpg.tar.7z', config.tmp_dir)
        unzip_tar('test-jpg.tar', config.tmp_dir)
        move_imgs('./test-jpg/*', config.original_jpg_test_image_path, config.tmp_dir)

    if not os.path.isfile(config.original_jpg_test_image_path + '/file_0.jpg'):
        puts(colored.red('downloading original sized additional test jpgs'))
        download_kaggle_file('test-jpg-additional.tar.7z', config.tmp_dir)
        unzip_7z('test-jpg-additional.tar.7z', config.tmp_dir)
        unzip_tar('test-jpg-additional.tar', config.tmp_dir)
        move_imgs('./test-jpg-additional/*', config.original_jpg_test_image_path, config.tmp_dir)

    puts(colored.cyan('full size jpgs ok'))
    return True

def validate_original_tif_test_images(tries):
    if tries == 0:
        puts(colored.cyan('failed to validate tif training images'))
        return False

    if not os.path.isfile(config.original_tif_test_image_path + '/test_0.tif'):
        puts(colored.red('downloading original test tifs'))
        download_kaggle_file('test-tif-v2.tar.7z', config.tmp_dir)
        unzip_7z('test-tif-v2.tar.7z', config.tmp_dir)
        unzip_tar('test-tif-v2.tar', config.tmp_dir)
        move_imgs('./test-tif-v2/*', config.original_tif_test_image_path, config.tmp_dir)

    puts(colored.cyan('full size tifs ok'))
    return True

def validate_train_csv(tries):

    if tries == 0:
        puts(colored.green('failed to create training csv'))
        return False

    ## CHECK IF FILE EXISTS
    if not os.path.isfile(config.train_csv_path):
        puts(colored.red('downloading training csv'))
        download_kaggle_file('train_v2.csv.zip', config.tmp_dir)
        remove_macosx('train_v2.csv.zip', config.tmp_dir)
        unzip_tar('train_v2.csv.zip', config.tmp_dir)
        move_from_tmp('train_v2.csv', config.tmp_dir, config.train_csv_path)
        return validate_train_csv(tries - 1)

    puts(colored.cyan('training csv ok'))
    return True

def download_kaggle_file(file, path):

    if not os.path.isfile(path + file):

        command('kg download -u {} -p {} -c {} -f {}'.format(
            credentials.login['username'],
            credentials.login['password'],
            credentials.login['competition'],
            file
        ), path, log=False)
    else:
        puts(colored.cyan('defaulting to local {}'.format(file)))


def unzip_7z(file, path):

    expected_name = file.replace(".7z", "")
    if not os.path.isfile(path + expected_name):
        puts(colored.cyan('* unzipping 7z'))
        command('7z x {} -aos'.format(file), path, log=False)
    else:
        puts(colored.cyan('defaulting to local {}'.format(expected_name)))


def unzip_tar(file, path):
    puts(colored.cyan('* unzipping Tar'))
    command('tar -zxvf {}'.format(file), path, log=False)

def remove_macosx(file, path):
    command('zip -d {} \"__MACOSX*\"'.format(file), path, log=False)

def move_from_tmp(file, path, new):
    puts(colored.cyan('* moving files'))
    command('mv {} {}'.format(file, new), path, log=False)

def move_imgs(source, target, path):
    puts(colored.cyan('* moving files'))
    command('for file in {}; do mv -- "$file" {} ; done'.format(source, target), path, log=False)


def clean_tmp():
    command('rm -r tmp/*', config.run_dir)


def launch():

    with indent(4):

        puts(colored.magenta('\ninitialising\n'))

        validate_directory_structure(directory_structure)

        validate_train_csv(1)

        validate_original_jpg_train_images(1)
        validate_original_tif_train_images(1)
        validate_original_jpg_test_images(1)
        validate_original_tif_test_images(1)


        puts('\n')
