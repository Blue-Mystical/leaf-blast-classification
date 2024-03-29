# leaf-blast-classification
Leaf Blast Classification using various classification models  
CSS496 Project  

The dataset of rice image is obtained from Kaggle. Only the healthy and rice blast rice dataset have been used for this project.  
https://www.kaggle.com/datasets/shayanriyaz/riceleafs  

# How to use:  
Running the first time: python main.py --dataset path/to/datasets  (this will take a while especially for bigger dataset size)
With resized folder: python main.py
Regenerate a resized folder: python main.py --forceresize --dataset path/to/datasets 

Arguments (if the argument is boolean, do not follow it with 'true'):    
--model [default: all]: Classification model to run  
 - available models: knn, rf, dt, svm, nb, all (the latter runs every models)  
--threads [default: -1]: Number of jobs used (-1 for all)  
--forceresize [default: False]: Whether the program should force regenerate the resized folder. the next argument --dataset pointing at a dataset folder is also required  
--dataset [required with --forceresize]: Path to the dataset folder when --forceresize is True  
--generatesize [default: -1]: How many images per subfolder are used to resize from the main dataset (Need forceresize = True & -1 means all image in each folder)  
--gridsearch [default: False]: If true, validates the best parameters from gridsearch.py instead of train_test normally  
--displayimage [default: False]: if true, show an image of a raw image and a HSV image for each folder (must press any key to progress)

