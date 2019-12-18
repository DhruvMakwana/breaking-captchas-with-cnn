# breaking-captchas-with-cnn

# Dependency
Required libraries and packages can be download from requirements.txt

`pip install -r requirements.txt`

# Dataset: 
we used `download_images.py` file to automatically download captcha images from url and save them in downloads folder. This file take single command line argument which is path to to the output directory

to run `download_images.py`, insert following command

`python download_images.py --output downloads`

# Labelling images

We used `annotate.py` file to label each digit from images in `downloads` images. For this task we have to press the key of digit which is appearing on screen while executing this file. This file take 2 command line argument --input is the path to our raw captcha images and --annot is path to where we’ll be storing the labeled digits

to run `annotate.py`, insert following command

`python annotate.py --input downloads --annot dataset`

# Training Model

We used `train_model.py` to train the network. This file takes two command line arguments --dataset is the path to the input dataset of labeled captcha digits (i.e., the dataset directory on disk) and --model is the path  to where our serialized LeNet weights will be saved after training

to run `train_model.py`, insert following command

`python train_model.py --dataset dataset --model output/lenet.hdf5`

# Testing Model

We used `test_model.py` to test the network. This file takes two command line arguments --dataset is the path to the input captcha images that we wish to break and --model is the  path to the serialized weights residing on disk

to run `test_model.py`, insert following command

`python test_model.py --input downloads --model output/lenet.hdf5`

