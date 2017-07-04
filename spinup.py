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

        fix_test_tifs()

        puts(colored.magenta('spectral'))
        create_spectral64(original_tif_train_image_path, spectral64_path + '/train')
        create_spectral64(original_tif_test_image_path, spectral64_path + '/test')

        create_visible(original_jpg_train_image_path, visible64_path + '/train', 64)
        create_visible(original_jpg_test_image_path, visible64_path + '/test', 64)

        create_visible(original_jpg_train_image_path, visible128_path + '/train', 128)
        create_visible(original_jpg_test_image_path, visible128_path + '/test', 128)

        create_visible(original_jpg_train_image_path, visible196_path + '/train', 196)
        create_visible(original_jpg_test_image_path, visible196_path + '/test', 196)





        puts('\n')