Heart Disease UCI
==============================

## Data

Download [data](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/download?datasetVersionNumber=2) and 
extract into folder `../datasets/heart.csv`

## Setup Virtual Environment
~~~
python -m venv heart_venv
source heart_venv/bin/activate
pip install -r requirements.txt
~~~

## Train
#### Logistic Regression
~~~
python heart/train.py hydra.job.chdir=True model=lr
~~~
#### Random forest
~~~ 
?
~~~

## Test:
~~~
pytest tests
~~~

## Linter:
~~~
pylint heart
~~~

## Project Organization
    ├── configs            <- Configuration files
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   ├── ?                   <- ?
    │   ├── entities            <- Configuration ORM entities
    │   ├── ?                   <- ?
    │   ├── models              <- Code to train models and then use trained models to make
    │   └── train.py   <- Script for training model
    │
    ├── artefacts          <- Hydra artefacts
    │   ├── ${now:%Y-%m-%d_%H-%M-%S}  <- Artefacts for every command
    │   │   ├── train.log             <- Train logs
    │   │   ├── models                <- Trained and serialized models, model predictions, or model summaries
    │   │   └── metrics               <- Models metrics
    │   └── ........................      
    ├── tests              <- unit & intagration tests
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.│
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                             generated with `pip freeze > requirements.txt`