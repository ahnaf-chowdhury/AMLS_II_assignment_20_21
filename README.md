# Sentiment Analysis on Movie Reviews

Classifies the sentiment of phrases from the Rotten Tomatoes dataset from 0 (very negative) to 5 (very positive).

## Requirements

The main packages used are Keras, Tensorflow, NLTK, scikit-learn, seaborn, pandas and NumPy. Use pip to install requirements and their dependencies using the 'requirements.txt' file.

```bash
pip install -r requirements.txt
```

## Usage

Compile and run main.py to train an LSTM and run it on a test set. (python=3.6)

```bash
python main.py
```

To use a pretrained model (trained using the same dataset and hyperparameter tunings) pass 'True' for the argument --pretrained (-p).

```bash
python main.py --pretrained True
```
or
```bash
python main.py -p True
```

## Files and Directories
The 'Datasets' directory contains the train and test datasets in the files 'train.tsv' adn 'test.tsv' respectively.

The directory 'A' contains other files related the task. The subdirectory 'A/models' contains a pretrained model in an h5 file and a csv file containing the history of training - metrics such as accuracy and loss. The subdirectory 'A/additional' contains python notebooks that are not integral to the task. They are only for reference to the work that has been done in obtaining the results for the task and tuning the hyperparameters.

The file ahnaf_nlp.py contains functions used by the main.py file.
