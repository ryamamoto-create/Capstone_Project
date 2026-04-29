# Project Structure
Capstone/
├── README.md
├── requirements.txt
├── .gitignore
├── .config
│
├── data/
│   ├── raw/
│   │   ├── data.txt
│   │   └── movieTitles.csv
│   └── processed/
│       └── ratings.parquet
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── parsing.py
│   ├── preprocessing.py
│   ├── split.py
│   ├── baseline.py
│   ├── model.py
│   ├── evaluation.py
│   └── utils.py
│
└── main.py