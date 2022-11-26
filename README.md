# leaf-blast-classification
(Work in progress)  
Leaf Blast Classification using KNN and other Models  
CSS496 Project  

The dataset is used from Kaggle:  
https://www.kaggle.com/datasets/shayanriyaz/riceleafs  

Only the healthy and rice blast rice dataset has been used.

# How to use:  
Run: python main.py --dataset Datasets  
Optional Arguments:  
--neighbors: K count in the classification system  
--threads: Number of thread used  
--forceresize: Whether the program should regenerate the resized folder  
--generatesize: How much image is sampled from the main dataset (Currently it is not randomized)

The first time it generates or when it is forced to generate a new folder with resized images will take a considerable amount of time.
