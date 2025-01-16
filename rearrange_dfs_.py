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
        pass

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
    args = parser.parser_args()


    start = args.start
    end = args.end
    test = args.test
    validation = args.validation
    training = args.training
    DATA_FRAME_SIZE = 5000
    data_path = "../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_en"

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
