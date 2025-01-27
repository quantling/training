import math
import pickle
import os
import pandas as pd
import gc
import psutil
import tqdm
import argparse
import logging


def rearrange_df(files, start, end, split):
    for file in tqdm(files):
        logging.info(f"Processing {file}")
        df = pd.read_pickle(f"{data_path}/{file}")

if __name__ == "main":


    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('-s', '--start', default=0) 
    parser.add_argument('-e', '--end' , default= float('inf'))
    parser.add_argument('--test', action='store_false')
    parser.add_argument('--validation', action='store_false')
    parser.add_argument('--training', action='store_false')
    parser.add_argument('--data_path', type=str, help="Path to the data directory w/o language.", default="../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder")
    parser.add_argument("--language", type=str, help="Language of the data.", default="en")
    parser.add_argument("--size", type=int, help="Size of the data.", default=5000)
    args = parser.parser_args()


    start = args.start
    end = args.end
    test = args.test
    validation = args.validation
    training = args.training
   
    

    sorted_files = sorted(os.listdir(data_path)) 

    training_files= [s for s in sorted_files if "training" in s.lower()]
    validation_files = [s for s in sorted_files if "validation" in s.lower()]
    test_files = [s for s in sorted_files if "test" in s.lower()]


    logging.info(f"Training files: {training_files}")
    logging.info(f"Validation files: {validation_files}")
    logging.info(f"Test files: {test_files}")

    if  not test:
        logging.info("Processing test files")
        rearrange_df(test_files, start, end, "test")
    if not validation:
        logging.info("Processing validation files")
        rearrange_df(validation_files, start, end, "validation")
    if not training:
        logging.info("Processing training files")
        rearrange_df(training_files, start, end, "training")
