# keel-bone-segmentation workflow
Here is the workflow to train the deep learning U-net model to 
segemtn the keel bone from the whole-body x-ray images of chickens   

#1 Please create a directory with name "keel-bone-segmentation", with the structure like this:
keel-bone-segmentation
├── data
│   ├── images
│   ├── masked
│   └── output
└── scripts
    ├── binary_keel_segmentation_learning_v1.py and binary_keel_segmentation_predicting_v1.py
    └── segmentation_functions.py ,  plot_across_folds_subplot.py and to_run_python3.txt

#2 Both "images" and "masked" are avaialbe to download from Zenodo link: https://doi.org/10.5281/zenodo.11172093 

#3 All codes are avialable to use from here: https://github.com/sallamslu/Keel-bone-segmentation 

#4 You can then run the training from "scripts" folder as following:
python3 binary_keel_segmentation_learning_v1.py   > ../data/output/binary_keel_segmentation_learning_v1.log   2>&1

#5 You can also run the the second script for visualizing original image with their true and predicted keel mask
python3 binary_keel_segmentation_predicting_v1.py > ../data/output/binary_keel_segmentation_predicting_v1.log 2>&1

#6 Optional, you can plot the training vs validation loss and dice in one figure by run
 "python3 plot_across_folds_subplot.py" in "../data/output/"

#7 You could run 4-6 by just run 
 "bash to_run_python3.txt" in the folder "../script"

#8 All results will be saved on "keel-bone-segmentation/data/output"


