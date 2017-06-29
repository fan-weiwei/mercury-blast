

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
    config.original_jpg_image_path,
    config.original_jpg_train_image_path
]

def validate_directory_structure(directories):
    try:
        head, *tail = directories

        if not head:
            puts(colored.green('directories ok'))
            return

        if not os.path.isdir(head):
            puts(colored.red('creating {}'.format(head)))
            command('mkdir {}'.format(head), config.run_dir)

        validate_directory_structure(tail)

    except ValueError: pass


def validate_original_jpg_train_images(tries):
    if tries == 0:
        puts(colored.green('failed to validate training images'))
        return False


    puts(colored.green('full size jpgs ok'))
    return True


def validate_train_csv(tries):

    if tries == 0:
        puts(colored.green('failed to create training csv'))
        return False

    ## CHECK IF FILE EXISTS
    file_exists = os.path.isfile(config.train_csv_path)
    if not file_exists:
        puts(colored.red('downloading training csv'))
        download_kaggle_file('train_v2.csv.zip', config.tmp_dir)
        remove_macosx('train_v2.csv.zip', config.tmp_dir)
        unzip_tar('train_v2.csv.zip', config.tmp_dir)
        move_from_tmp('train_v2.csv', config.tmp_dir, config.train_csv_path)
        clean_tmp()
        return validate_train_csv(tries - 1)

    puts(colored.green('training csv ok'))
    return True


def download_kaggle_file(file, path):

    command('kg download -u {} -p {} -c {} -f {}'.format(
        credentials.login['username'],
        credentials.login['password'],
        credentials.login['competition'],
        file
    ), path, log=False)


def unzip_7z(file, path):
    command('7z x {}'.format(file), path)


def unzip_tar(file, path):
    command('tar -zxvf {}'.format(file), path, log=False)


def remove_macosx(file, path):
    command('zip -d {} \"__MACOSX*\"'.format(file), path, log=False)

def move_from_tmp(file, path, new):
    command('mv {} ../{}'.format(file, new), path, log=False)

def clean_tmp():
    command('rm -r tmp/*', config.run_dir)

def launch():

    with indent(4):

        puts(colored.magenta('\ninitialising\n'))
        validate_directory_structure(directory_structure)
        validate_train_csv(5)
        validate_original_jpg_train_images(5)



