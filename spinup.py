

## TODO Check for existance of full scale images
## TODO Report diagnostic
## TODO If full scale jpegs don't exist redownload
## TODO Check fo existance of tifs (For NIR)
## TODO If tifs don't exist redownload

from clint.textui import colored, puts, indent


'''
    if arguments['validate']:
        with indent(4):
            puts(colored.magenta('\nAll good\n'))
            puts('hello world\n\n')


            img_resize = (16, 16)
        color_channels = 3  # RGB
        train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()

        assert os.path.exists(train_jpeg_dir), "The {} folder does not exist".format(train_jpeg_dir)
        assert os.path.exists(test_jpeg_dir), "The {} folder does not exist".format(test_jpeg_dir)
        assert os.path.exists(train_csv_file), "The {} file does not exist".format(test_jpeg_additional)
        assert os.path.exists(train_csv_file), "The {} file does not exist".format(train_csv_file)

        x_train, x_test, y_train, y_map, x_test_filename = data_helper.preprocess_data(train_jpeg_dir, test_jpeg_dir,
                                                                                       test_jpeg_additional,
                                                                                       train_csv_file, img_resize)
        labels_df = pd.read_csv(train_csv_file)
        labels_count = len(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
        train_files_count = len(os.listdir(train_jpeg_dir))
        test_files_count = len(os.listdir(test_jpeg_dir)) + len(os.listdir(test_jpeg_additional))
        assert x_train.shape == (train_files_count, *img_resize, color_channels)
        assert x_test.shape == (test_files_count, *img_resize, color_channels)
        assert y_train.shape == (train_files_count, labels_count)
'''

import os
import config
import credentials
import subprocess

def validate_train_csv():

    file_exists = os.path.exists(config.train_csv_path)
    return file_exists

def download_kaggle_file(file, path):

    command = 'kg download -u {} -p {} -c {} -f {}'.format(
        credentials.login['username'],
        credentials.login['password'],
        credentials.login['competition'],
        file
    )

    subprocess.call([command], shell=True, cwd=path)

def unzip_7z(file, path):
    command = '7z x {}'.format(file)
    subprocess.call([command], shell=True, cwd=path)


def unzip_tar(file, path):
    command = 'tar -zxvf {}'.format(file)
    subprocess.call([command], shell=True, cwd=path)



def launch():

    with indent(4):

        puts(colored.magenta('\nInitialising\n'))

        if validate_train_csv:
            puts(colored.green('training csv ok'))
        else:
            puts(colored.red('training csv missing -> downloading'))

        #download_kaggle_file('train_v2.csv.zip', config.tmp_dir)
        unzip_tar('train_v2.csv.zip', config.tmp_dir)


