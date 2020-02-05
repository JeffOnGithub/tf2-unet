CONFIG = dict()

CONFIG['DATASET_VIRTUAL'] = dict()
CONFIG['DATASET_REAL'] = dict()
CONFIG['DATASET_REAL_TEST'] = dict()
CONFIG['DATASET_GI_AGRICUTURE'] = dict()
CONFIG['DATASET_SOYA_CORN'] = dict()

CONFIG['MODEL'] = dict()
CONFIG['TRAINING'] = dict()

CONFIG['TRAINING']['EPOCHS'] = 50
CONFIG['TRAINING']['VAL_SUBSPLITS'] = 2

########################
# DATASET VIRTUAL
########################

CONFIG['DATASET_VIRTUAL']['NATIVE_HEIGHT'] = 1080
CONFIG['DATASET_VIRTUAL']['NATIVE_WIDTH'] = 1920
CONFIG['DATASET_VIRTUAL']['SIZE_HEIGHT'] = 448
CONFIG['DATASET_VIRTUAL']['SIZE_WIDTH'] = 448
CONFIG['DATASET_VIRTUAL']['N_CLASSES'] = 8
CONFIG['DATASET_VIRTUAL']['BATCH_SIZE'] = 14
CONFIG['DATASET_VIRTUAL']['BATCH_FACTOR'] = 20

CONFIG['DATASET_VIRTUAL']['train_imgs_folder'] = "Z:/Unreal_Datasets/unrealmeadow/Images_Categorical/*.jpg"
CONFIG['DATASET_VIRTUAL']['test_imgs_folder'] = "Z:/Unreal_Datasets/unrealmeadow/Images_Categorical/*.jpg"
CONFIG['DATASET_VIRTUAL']['img_folder_path'] = "Images_Categorical"
CONFIG['DATASET_VIRTUAL']['lbl_folder_path'] = "Labels_Categorical"
CONFIG['DATASET_VIRTUAL']['img_ext'] = ".jpg"
CONFIG['DATASET_VIRTUAL']['lbl_ext'] = ".png"

########################
# DATASET REAL
########################

CONFIG['DATASET_REAL']['NATIVE_HEIGHT'] = 3000
CONFIG['DATASET_REAL']['NATIVE_WIDTH'] = 4000
CONFIG['DATASET_REAL']['SIZE_HEIGHT'] = 448
CONFIG['DATASET_REAL']['SIZE_WIDTH'] = 448
CONFIG['DATASET_REAL']['N_CLASSES'] = 8
CONFIG['DATASET_REAL']['BATCH_SIZE'] = 14
CONFIG['DATASET_REAL']['BATCH_FACTOR'] = 20

CONFIG['DATASET_REAL']['train_imgs_folder'] = "Z:/Real_Datasets/drone-lac/Images_Categorical/*.JPG"
CONFIG['DATASET_REAL']['test_imgs_folder'] = "Z:/Real_Datasets/drone-lac/Images_Categorical/*.JPG"
CONFIG['DATASET_REAL']['predict_imgs_folder'] = "Z:/Real_Datasets/drone-lac/Images_Predict/*.JPG"
CONFIG['DATASET_REAL']['img_folder_path'] = "Images_Categorical"
CONFIG['DATASET_REAL']['lbl_folder_path'] = "Labels_Categorical"
CONFIG['DATASET_REAL']['img_ext'] = ".JPG"
CONFIG['DATASET_REAL']['lbl_ext'] = "_watershed_mask.png"

########################
# DATASET REAL TEST SET
########################

CONFIG['DATASET_REAL_TEST']['NATIVE_HEIGHT'] = 3000
CONFIG['DATASET_REAL_TEST']['NATIVE_WIDTH'] = 4000
CONFIG['DATASET_REAL_TEST']['SIZE_HEIGHT'] = 448
CONFIG['DATASET_REAL_TEST']['SIZE_WIDTH'] = 448
CONFIG['DATASET_REAL_TEST']['N_CLASSES'] = 8
CONFIG['DATASET_REAL_TEST']['BATCH_SIZE'] = 14
CONFIG['DATASET_REAL_TEST']['BATCH_FACTOR'] = 20

CONFIG['DATASET_REAL_TEST']['train_imgs_folder'] = "Z:/Real_Datasets/drone-lac/Images_Train/*.JPG"
CONFIG['DATASET_REAL_TEST']['test_imgs_folder'] = "Z:/Real_Datasets/drone-lac/Images_Test/*.JPG"
CONFIG['DATASET_REAL_TEST']['img_folder_path'] = "Images_Test"
CONFIG['DATASET_REAL_TEST']['lbl_folder_path'] = "Labels_Test"
CONFIG['DATASET_REAL_TEST']['img_ext'] = ".JPG"
CONFIG['DATASET_REAL_TEST']['lbl_ext'] = "_watershed_mask.png"

########################
# DATASET GOOGLE IMAGES AGRICULTURE
########################

CONFIG['DATASET_GI_AGRICUTURE']['NATIVE_HEIGHT'] = 1080
CONFIG['DATASET_GI_AGRICUTURE']['NATIVE_WIDTH'] = 1920
CONFIG['DATASET_GI_AGRICUTURE']['SIZE_HEIGHT'] = 448
CONFIG['DATASET_GI_AGRICUTURE']['SIZE_WIDTH'] = 448
CONFIG['DATASET_GI_AGRICUTURE']['N_CLASSES'] = 8
CONFIG['DATASET_GI_AGRICUTURE']['BATCH_SIZE'] = 14
CONFIG['DATASET_GI_AGRICUTURE']['BATCH_FACTOR'] = 20

CONFIG['DATASET_GI_AGRICUTURE']['train_imgs_folder'] = "Z:/Real_Datasets/gi-agriculture/*.jpg"
CONFIG['DATASET_GI_AGRICUTURE']['test_imgs_folder'] = "Z:/Real_Datasets/gi-agriculture/*.jpg"
CONFIG['DATASET_GI_AGRICUTURE']['img_folder_path'] = "gi-agriculture"
CONFIG['DATASET_GI_AGRICUTURE']['lbl_folder_path'] = "gi-agriculture"
CONFIG['DATASET_GI_AGRICUTURE']['img_ext'] = ".jpg"
CONFIG['DATASET_GI_AGRICUTURE']['lbl_ext'] = ".jpg"

########################
# DATASET SOYA CORN
########################

CONFIG['DATASET_SOYA_CORN']['NATIVE_HEIGHT'] = 2160
CONFIG['DATASET_SOYA_CORN']['NATIVE_WIDTH'] = 3840
CONFIG['DATASET_SOYA_CORN']['SIZE_HEIGHT'] = 448
CONFIG['DATASET_SOYA_CORN']['SIZE_WIDTH'] = 448
CONFIG['DATASET_SOYA_CORN']['N_CLASSES'] = 8
CONFIG['DATASET_SOYA_CORN']['BATCH_SIZE'] = 14
CONFIG['DATASET_SOYA_CORN']['BATCH_FACTOR'] = 20

CONFIG['DATASET_SOYA_CORN']['train_imgs_folder'] = "Z:/Real_Datasets/soya-corn/*.jpg"
CONFIG['DATASET_SOYA_CORN']['test_imgs_folder'] = "Z:/Real_Datasets/soya-corn/*.jpg"
CONFIG['DATASET_SOYA_CORN']['img_folder_path'] = "soya-corn"
CONFIG['DATASET_SOYA_CORN']['lbl_folder_path'] = "soya-corn"
CONFIG['DATASET_SOYA_CORN']['img_ext'] = ".jpg"
CONFIG['DATASET_SOYA_CORN']['lbl_ext'] = ".jpg"