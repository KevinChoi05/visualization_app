D-Prime Value Prediction for Visualizations


Overview
This repository contains a machine learning model designed to predict the D-prime value of visualizations. The D-prime value is a metric that quantifies the memorability of visualizations, calculated using the hit rate and false alarm rate.


Dataset
The model is trained on a dataset that is pivotal to research in the field of data visualization memorability. The dataset is based on research work presented in the following papers:


"What Makes a Visualization Memorable?" by Borkin et al.
"Beyond Memorability: Visualization Recognition and Recall" by Borkin et al.


How to Obtain the Dataset
Visit the Massachusetts Visualization (MassVis) dataset page.
Scroll down to the "Download the Dataset" section.
Fill out the request form to download the dataset.
By downloading and using this dataset, you agree to use it only for research and educational purposes. Commercial use or distribution is not permitted.

LINK-----> http://massvis.mit.edu/



Data Files

targets393_metadata.csv: Contains metadata and attributes of each visualization.

targets393: 393 visualizations


Model
The model is a multi-input neural network that takes both image data and metadata as inputs. It comprises convolutional layers for image feature extraction and dense layers for metadata feature extraction.
The two pathways are then concatenated and processed through additional dense layers to predict the D-prime value.
