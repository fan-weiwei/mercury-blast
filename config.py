import os


run_dir = '/Users/fanweiwei/rainforest/mercury-blast'

tmp_dir = run_dir + '/tmp/'
training_data_dir = run_dir + '/training-data'
train_csv_path = training_data_dir + '/train.csv'



#######################################################

images_path = run_dir + '/images'

original_jpg_image_path = images_path + '/original-jpg'
original_tif_image_path = images_path + '/original-tif'

small_train_image_path = run_dir + '/images/small/train'
small_test_image_path  = run_dir + '/images/small/test'

medium_train_image_path = run_dir + '/images/medium/train'
medium_test_image_path  = run_dir + '/images/medium/test'

original_jpg_train_image_path = original_jpg_image_path + '/train'
original_tif_train_image_path = original_tif_image_path + '/train'

original_jpg_test_image_path = original_jpg_image_path + '/test'
original_tif_test_image_path = original_tif_image_path + '/test'

#######################################################

spectral64_path = images_path + '/spectral64'





#######################################################


sample_submission_path = run_dir + '/submissions/sample_submission.csv'
