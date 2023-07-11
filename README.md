# CRCP-a-user-friendly-CRC-risk-prediction-system
Exploring Hybrid Indexes of Complete Blood Count for Improving Cost-effective Colorectal Cancer Prediction

Required packages or libraries
python=3.8.8
joblib=1.0.1
lightgbm=3.3.3
matplotlib=3.2.1
numpy=1.18.2
pandas=1.2.4
scipy=1.6.2
seaborn=0.10.0
shap=0.40.0
xgboost=1.5.1

r=4.1.1
ggplot2=3.4.2
reshape2=1.4.4
magick=2.7.3

The script provided in this GitHub account should be executed in that particular order to generate the analysis results in the manuscript.

Please use the data provided as samples in the manuscript ('data.csv' and 'independent testset.csv') as the corresponding input. The content of the data can be modified when using the code.

Description
The packages to be installed and information about them are contained in the package_list.txt file.
First modify the dataset you are using in the data.csv format in the dataset folder with the following column names: 'label', 'age', 'sex', 'RBC', 'HCT', 'HGB', 'RDW-CV', 'PLT', 'PCT', 'PDW', 'WBC', 'MON#', 'NEU#', ' LYM#','BAS#','EOS#', excluding labels, for a total of 15 whole blood count features. independent testset.csv was modified in the appropriate format.
To fuse the data with whole blood cell features, run the Feature_Fusion.py file to obtain the fused data fusion.csv and independent testset fusion.csv files.
Next, run the First.py file to perform the dataset partitioning, normalisation and feature importance ranking, with the results stored in the normalisation folder. The default here is to use the non-fused data data.csv and independent testset.csv for the demonstration.
Next, run Data_Structure.R to get the feature structure of the positive and negative samples of the training set, get the results and save them in the results folder.
Statistical_Analysis.py is then run to get the statistics of the training set, including p-values and Pearson correlation coefficients. And pre-process the feature structure for the r-language implementation of positive and negative samples.
Finally run the Second.py file to build the model, get the results and save them in the results folder.
