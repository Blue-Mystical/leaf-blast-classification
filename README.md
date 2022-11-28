# leaf-blast-classification
(Work in progress)  
Leaf Blast Classification using various classification models  
CSS496 Project  

The dataset is used from Kaggle:  
https://www.kaggle.com/datasets/shayanriyaz/riceleafs  

Only the healthy and rice blast rice dataset has been used.

# How to use:  
Run: python main.py --dataset Datasets  

Arguments:  
--dataset [REQUIRED]: Path to the dataset folder  
--model [default: all]: Classification model to run  
 - available models: knn, rdf, dt, svm, nb, all (run every models)  
--threads [default: -1]: Number of threads used (-1 for all)  
--forceresize [default: False]: Whether the program should force regenerate the resized folder  
--generatesize [default: -1]: How many images per subfolder is used to resize from the main dataset (Need forceresize = True & -1 means all image)  
--gridsearch [default: False]: If true, validates the best parameters from gridsearch.py instead of train_test normally  
--displayimage [default: False]: if true, show an image of a raw image and a HSV image for each folder (must press any key to progress)

The first time it generates or when it is forced to generate a new folder with resized images will take a considerable amount of time.
