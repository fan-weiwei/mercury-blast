import os

tmp_dir = './tmp/'
training_data_path = './training-data'
train_csv_path = training_data_path + '/train.csv'

run_dir = os.getcwd()



#######################################################

images_path = './images'

original_jpg_image_path = images_path + '/original-jpg'

small_train_image_path = './images/small/train'
small_test_image_path  = './images/small/test'

medium_train_image_path = './images/medium/train'
medium_test_image_path  = './images/medium/test'

original_jpg_train_image_path = original_jpg_image_path + '/train'

original_test_image_path_one  = './images/original/test'
original_test_image_path_two  = './images/original/test-jpg-additional'

#######################################################


sample_submission_path = './submissions/sample_submission.csv'
