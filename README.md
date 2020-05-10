# w4761-final_project
## Overview
Repository for our final project for Computational Genomics (COMS W4761)

## Files
* main.py  
This the main function to take the input file, preprocess the data, tokenize, and train the model.
* cnn.py  
This file is used to configure the CNN. 
* rnn.py  
This file is used to configure the RNN. 
* tokenizer.py  
This file contains functions for pre-processing the data such as purging the input data, tokenize the RNA sequence, and creating features and labels for training. 
* post_process.py  
This file contains functions for plotting the training vs validation loss and accuracy.
* RNA_sequence_input.csv  
This the input file for the main function

## Files in folder dataset_preparation_Kevin_Wong_hw2735
* Cleaning up ct files.ipynb
This Jupyter Notebook cleans the .ct files initially downloaded from RNA STRAND database, and also contains scripts to integrate the converted .ct files into one single .csv file in dot-bracket notation. Further explanation is given within the notebook
* RNA_secondary_structure.csv
This is the completed dataset for the project
* cmd commands.xlsx
Excel is used to create the commands for "ct2dot" tool
* ct2dot_cmd.txt
This is the command line script used to execute the"ct2dot" tool, within Windows environment

## Files in folder rnng_code