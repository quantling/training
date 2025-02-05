import math
import pickle
import os
import pandas as pd
import gc
import psutil
import tqdm
import argparse
import logging
import re
import shutil 


from split_utils import NotEnoughDiskSpaceError

def rearrange_df(files, start, end, split,size, data_path, output_path, identifier, delete=False):
    former = []
    already_written = []
    big_df = pd.DataFrame()
    if end == -1:
         files = files[start:]
        
    else:
        files = files[start:end]
    
    i = start
    og_length = len(files) 
    while len(files) > 0:

        logging.info(f"Files left: {len(files)} of {og_length} files at the beginning")

        total, used, free = shutil.disk_usage(data_path)

        logging.debug("Total: %d GiB" % (total // (2**30)))
        logging.debug("Used: %d GiB" % (used // (2**30)))
        logging.info("Free: %d GiB" % (free // (2**30)))
        

        if free < 2**30 *24:
            raise NotEnoughDiskSpaceError(f"Free disk space is less than 24 GiB. Free: {free // (2**30)} GiB")
            
        if len(big_df.index) < size:
            file = files.pop(0)
            logging.info(f"Processing {file}")
           
            from_file = pd.read_pickle()
            former.append((file, len(from_file.index),from_file, i)) # Since I want to keep track of the files I have processed because we might need to delete them later

            big_df = pd.concat([big_df, from_file], ignore_index=True)
        logging.info(f"Current size of big_df: {len(big_df.index)}")

        if len(big_df.index) >= size:
            logging.info(f"Size of big_df exceeds {size}. Trimming.")
            to_write = big_df[:size]
            logging.info(f"Size of big_df after trimming: {len(big_df.index)}")
            to_write.to_pickle(f"{output_path}/{split}_{identifier}_{i}.pkl")
            already_written.append(to_write)
            i += 1
            logging.info(f"Written {split} file {i}")

        if delete:
            oldest_file  = former[0]
            files_passed = i - oldest_file[3]
            if oldest_file[1] <= (size * (files_passed)): #only if the file could have been written 
            # we need to check if all the words have been written
                candidates = former[:files_passed]
                check_df = pd.concat(candidates, ignore_index=True)
                oldest_file_subset = set(oldest_file[2]["lexical_word"].unique()) #I chose lexical word because it tends to be more unique than the label
                check_df_subset = set(check_df["lexical_word"].unique())
                if oldest_file_subset.issubset(check_df_subset):
                    logging.info(f"Deleting {oldest_file[0]}")
                    os.remove(oldest_file[0])
                    former.pop(0)
                    
                    former = former[files_passed -1:]
                else:
                    logging.info(f"Cannot delete {oldest_file[0]} because not all words have been written")
        else:
            logging.info("Not deleting files")
            former = []
            already_written = []
            gc.collect()

    logging.info("Writing the last file")

    total, used, free = shutil.disk_usage(data_path)

        
        

    if free < 2**30 *24:
            raise NotEnoughDiskSpaceError(f"Free disk space is less than 24 GiB. Free: {free // (2**30)} GiB")
    big_df.to_pickle(f"{output_path}/{split}_{identifier}_{i}.pkl")
    logging.info(f"Written {split} file {i}")
    logging.info("Done")
                

        

if __name__ == "main":
       
    logger = logging.getLogger("file_logger")
    logger.setLevel(logging.INFO)

    
    file_handler = logging.FileHandler("file_lists.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

   
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s")) # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')


    parser.add_argument('--test', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--data_path', type=str, help="Path to the data directory w/o language.", default="../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder")
    parser.add_argument("--language", type=str, help="Language of the data.", default="en")
    parser.add_argument("--size", type=int, help="Size of the data.", default=5000)
    parser.add_argument("--output_path", type=str, help="Path to the output directory.", default="../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder")
    parser.add_argumennt("--ingore", type=str, help="Ignore files with this Regex.", default="")
    parser.add_argument("--identifier", type=str, help="Identifier for the output files.", default="rearranged")
    parser.add_argument("--start_test", type=int, help="Start index for test set.", default=0)
    parser.add_argument("--end_test", type=int, help="End index for test set.", default=-1)
    parser.add_argument("--start_validation", type=int, help="Start index for validation set.", default=0)
    parser.add_argument("--end_validation", type=int, help="End index for validation set.", default = -1)
    parser.add_argument("--start_training", type=int, help="Start index for training set.", default=0)
    parser.add_argument("--end_training", type=int, help="End index for training set.", default=-1)
    parser.add_argument("--delete", action="store_false", help="Delete the original files after processing.")


    args = parser.parser_args()


    start = args.start
    end = args.end
    test = args.test
    validation = args.validation
    training = args.training
    data_path = args.data_path +  "_" +args.language
    

    sorted_files = sorted(os.listdir(data_path)) 
    ignore = args.ignore
    if args.ignore != "":
        logging.info("Ignoring files with the following ReGex:", ignore)
        files = [s for s in sorted_files if re.search(ignore, s) is None]
        logging.info("These are the files after applying the ReGex:", files)
    else:
        files = sorted_files

    training_files = [s for s in sorted_files if "training" in s.lower()]
    validation_files = [s for s in sorted_files if "validation" in s.lower()]
    test_files = [s for s in sorted_files if "test" in s.lower()]

    
    def log_file_list(name, file_list):
        log_message = f"{name} files:\n" + "\n".join(f"{i}: {s}" for i, s in enumerate(file_list))
        logger.info(log_message)
    log_file_list("Training", training_files)
    log_file_list("Validation", validation_files)
    log_file_list("Test", test_files)


    if  not test:
        logging.info("Processing test files")
        rearrange_df(test_files, args.start_test, args.end_test, "test",args.size, data_path, args.output_path, args.identifier, args.delete) 
    if not validation:
        logging.info("Processing validation files")
        rearrange_df(validation_files, args.start_validation, args.end_validation, "validation", args.size, data_path, args.output_path, args.identifier, args.delete)
    if not training:
        logging.info("Processing training files")
        rearrange_df(training_files, args.start_validation, args.end_validation, "training", args.size, data_path, args.output_path, args.identifier, args.delete)
