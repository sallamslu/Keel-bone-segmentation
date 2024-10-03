# keel-bone-segmentation workflow
Here is the workflow to train the deep learning U-net model to 
segment the keel bone from the whole-body x-ray images of chickens   

#1 Please create a directory with name "keel-bone-segmentation", with the structure like this:

![directory_structure](https://github.com/sallamslu/Keel-bone-segmentation/assets/91287246/399f8b9d-5482-4318-aa40-9e47181665e8)


#2 Both "images" and "masked" are avaialbe to download from Zenodo link: https://doi.org/10.5281/zenodo.11172093 

#3 All scripts are avialable to use here: https://github.com/sallamslu/Keel-bone-segmentation 

#4 Run the training from "scripts" folder as following:
python3 binary_keel_segmentation_learning_v1.py > ../data/output/binary_keel_segmentation_learning_v1.log   2>&1

#5 You can visualize the original image with their true and predicted keel masks:
python3 binary_keel_segmentation_predicting_v1.py > ../data/output/binary_keel_segmentation_predicting_v1.log 2>&1

#6 Optional, you can plot the training vs validation loss and dice across 5 folds in one figure by run:
"python3 plot_across_folds_subplot.py" in "../data/output/"

#7 You could run 4-6 by just run:
"bash to_run_python3.txt" in the folder "../script"

#8 All results will be saved on "keel-bone-segmentation/data/output"


