# IR Project 1

For the course "Information Retrieval and Extraction", Fall semester, 2017.
Writen by Hao-Yung Chan (Student ID: R04521618)

## Requirements
1. Python 3.6 (3.6.1 is used in development)
2. virtualenv (pyenv-virtualenv is OK)

## Installation
For macOS and Linux:
```
  $ git clone https://github.com/katrina376/ir2017-prj1.git
  $ cd ir2017-prj1
  $ virtualenv -p python3 venv
  $ source venv/bin/activate
  $ pip install -r requirements.txt
  $ python -m nltk.downloader stopwords
  $ python -m nltk.downloader punkt
```

For Windows:
```
  $ git clone https://github.com/katrina376/ir2017-prj1.git
  $ cd ir2017-prj1
  $ virtualenv -p python3 venv
  $ venv\Scripts\activate
  $ pip install -r requirements.txt
  $ python -m nltk.downloader stopwords
  $ python -m nltk.downloader punkt
```

## Run
Before running the scripts, put `DBdoc.json` and `queries-v2.txt` which are provided by the course right inside the project directory.
```
  $ python main.py
```
The result will be saved as `result-file.txt`.
