# brainhack-EEG-depression-ml-dl
Brain Hack School 2025 Project | Decoding Depression from EEG using ML (SVM) and DL (EEGNet) 

---
type: "project" # DON'T TOUCH THIS ! :)

date: "2025-06-15"

title: "Decoding Depression via EEG Biomarkers: A Neurocomputational Approach using Machine and Deep Learning"

names: [Wei-Xuan Chai]

github_repo: [https://github.com/PSY6983-2021/project_template](https://github.com/ChaiWeiXuan/brainhack-EEG-depression-ml-dl)

tags: [EEG, depression, svm, eegnet]

summary: "This project was conducted as part of Brainhack School 2025. It aimed to classify major depressive disorder (MDD) using temporal-domain EEG features (i.e., band power), applying both machine learning (SVM) and deep learning (EEGNet) models."



## Project definition

### Background

This project was inspired by a clinical case from Shanghai Ruijin Hospital, where Brain-Computer Interface (BCI) technology was applied to treat severe depression. That real case made me wonder about the potential of neural decoding in understanding and supporting mental health.

As someone interested in both neuroscience and BCI, I believe it is essential to master machine learning and deep learning techniques—especially when working with high-dimensional signals like EEG or other neuroimaging data. This project is my personal attempt to explore how machine learning (SVM) and deep learning (EEGNet) models perform in classifying major depressive disorder (MDD) based on EEG band power features.


### Tools

The project will rely on the following technologies:

1. Programming & Environment
- Python
- Jupyter Notebook: Used for data preprocessing and training the SVM machine learning model.
- Google Colab: Used for training the EEGNet deep learning model.

2. Libraries & Frameworks
- scikit-learn, gridsearchCV: For machine learning models.
- MNE: For EEG data processing and visualization.
- numpy, pandas, matplotlib: For data manipulation and plotting.
- tensorflow, keras: For deep learning models.

3. External Models / Resources
- EEGNet implementation from ARL (Army Research Laboratory) EEGModels Project [page](https://github.com/vlawhern/arl-eegmodels)
- Brainhack School Modules [page]((https://school.brainhackmtl.org/))


### Data
This project uses the MDD Patients and Healthy Controls EEG Data (New), downloaded from [page](https://figshare.com/articles/dataset/EEG_Data_New/4244171/2).
- The dataset includes 30 Healthy Controls (H) and 34 Major Depressive Disorder (MDD) participants.
- Each participant completed three conditions:
  - Eyes Closed (EC)
  - Eyes Open (EO)
  - P300 Task (TASK)

For this analysis, we used:
- Only the 5-minute EC EEG data
- Successfully downloaded: 28 H and 30 MDD subjects


### Deliverables

At the end of this project, we will have:
 - 1. Preprocessing & Bandpower Feature.ipynb
 - 2. Machine Learning_SVM.ipynb
 - 3. Deep Learning_EEGNet.ipynb
These deliverables reflect the full pipeline from raw EEG preprocessing to both traditional ML and DL-based classification approaches.


## Results

### Progress overview

This project started with the goal of integrating what I’ve previously learned—EEG preprocessing, machine learning, and deep learning—into a single end-to-end pipeline. While I had prior experience working with EEG data and individual modeling techniques, I had never tried building an entire classification workflow on my own, from data preparation to model comparison.

I began by preprocessing the EEG data using MNE-Python and extracting temporal-domain features (band power). Then I implemented an SVM classifier, followed by a deep learning model based on EEGNet. Although the EEGNet part was more challenging due to model tuning and GPU limitations, I still managed to get a working prototype on Google Colab.

Beyond modeling, I also learned how to structure a project repository, push updates to GitHub, and write clear documentation—sometimes after a few frustrating errors (and moments of panic). But overall, it was a fun and fulfilling learning experience, and a solid first attempt at decoding mental health signals using both machine learning and deep learning techniques.

### Tools I learned during this project

- Learned to preprocess EEG signals and extract bandpower features using MNE.
- Applied classical machine learning models (SVM) with scikit-learn and gridsearchCV.
- Implemented and trained a deep learning EEG model (EEGNet) with TensorFlow and Keras.
- Developing, testing, and executing the entire analysis pipeline using Jupyter Notebooks and Google Colab as interactive environments


### Results

Preliminary results suggested that SVM achieved higher accuracy (> 0.95) compared to EEGNet (~0.8), possibly due to limited hyperparameter tuning and model optimization in this early-stage exploration.

#### Deliverable 1: Preprocessing & Bandpower Feature

EEG preprocessing pipeline using MNE-Python, and extraction of frequency-domain features (band power) for classification.

#### Deliverable 2: Machine Learning_SVM

Training and evaluation of a Support Vector Machine (SVM) model based on extracted band power features.  

#### Deliverable 3: Deep Learning_EEGNet

Training and evaluation of a EEGNet model using preprocessing EEG epoch data, implemented on Google Colab with GPU support.


## Conclusion and acknowledgement


