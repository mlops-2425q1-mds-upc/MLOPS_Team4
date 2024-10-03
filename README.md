# MLOps_project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Sentimental Analysis applied in Twitter

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Dataset and model cards
│
├── models             <- Scripts to train and preprocess data for models used for │                         experiments in Milestone 1
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Sentiment_Analysis and configuration for tools like black
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   │
│   └── figures        <- Generated graphics and figures to be used in reporting
│   │
│   └── logistic_regression_model       <- figs from LR
│   │
│   └── lstm_model       <- figs from lstm model
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── Sentiment_Analysis   <- Source code for use in this project.
    │
    ├── config.py             <- Store useful variables and configuration
    │
    ├── clean.py              <- Pocess raw dataset
    │
    ├── data_preprocessing    <- Adapt processed dataset to LSTM
    │
    └── LSTM_training.py      <- Code to train LSTM
```

--------

