from pull import *
from clint.textui import *
from preprocessor import *


def run():

    with indent(4):

        puts(colored.magenta('\ninitializing'))

        initial_check()
        validate_train_csv()
        validate_mapping_csv()
        validate_images()

        puts(colored.magenta('spectral'))
        create_spectral64(original_tif_train_image_path, spectral64_path + '/train')
        create_spectral64(original_tif_test_image_path, spectral64_path + '/test')



        puts('\n')