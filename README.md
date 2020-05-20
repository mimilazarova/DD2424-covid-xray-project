# DD2424-covid-xray-project

This repo is the home of the project "Detection of COVID-19 Based on Radiography Images and Identification of the Responsible Regions." It is the group project for the course DD2424 Deep Learning in Data Science, given in spring 2020.

The work would not have been possible without the COVIDx dataset. This dataset is publically avaliable at https://github.com/lindawangg/COVID-Net. The version used here is COVIDx2, published on 15th of April.

It is important to be noted that any labaling by our model is not a diaginosis.

In the repository one can find the following files:

preprocess.py - this implements a class for loading images to np.arrays
To preproces the data, one has to download COVIDx following the instructionsin the link above. After that, change the path to the data folder, as well as to the train.txt and test.txt file, in the main function of preprocess.py and run it.

CNN_COVID19.ipynb - in this notebook one can see the architecture of the final model, as well as some of the experiments we tried in order to increase the accuracy of the model.

Grad_CAM.ipynb - here is the code for the explaynability method Grad-CAM.

SHAP_LIME.ipynb - here is the code for thr explainability methods SHAP and LIME.

covid_explainability.ipynb - is an aggregate notebook for all explainability methods.

