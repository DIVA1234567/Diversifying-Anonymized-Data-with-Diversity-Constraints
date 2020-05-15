import re
import sys
import pandas as pd 

# Drug Dataset FDs 
FD_INPUT_INFO = [[3, 4], [5, 4], [6, 7]]
FD_OUTPUT_INFO = [[7, 5], [4, 5], [1, 2]]
# FD_INPUT_INFO = [[0, 1]]
# FD_OUTPUT_INFO = [[0, 1]]



# Retrieve the pattern information from the dataset (FD-value based)
def retrieve_pattern_info(dataset_name, FD_INFO):
     # print("##########################  Retrieve pattern info")
     pattern_info = set() 
     match = re.search(r'data/', dataset_name)
     if match is None:
          dataset_name = "data/" + dataset_name
     print("#####################     %s     ####################" %(dataset_name))
     # Open the dataset in read only mode
     with open(dataset_name, 'r') as in_file:
          # Read the dataset information one line at a time
          for num, line in enumerate(in_file, start=0):
               # Strip any trailing or leading spaces from line read from file
               strip_line = line.strip()
               # Split the line read from file (generates a list)
               split_line = strip_line.split(',')
               # print(num, split_line)
               # Iterate through the FDs to generate keys to store pattern information
               for item in FD_INFO:
                    pattern_key = None
                    for i in range(len(item)):
                         if pattern_key is None:
                              pattern_key = split_line[item[i]]
                         else:
                              pattern_key = pattern_key + " " + split_line[item[i]]
                    # Add the pattern to the set 
                    pattern_info.add(pattern_key)
     # print("Pattern info :", pattern_info)
     print("Pattern info count :", len(pattern_info))
     print("Patterns :")
     for _key in pattern_info:
          print("%s" %(_key))
     print("\n\n")
     return pattern_info
     # print("#################################################")



if __name__ == '__main__':
     # Parse the command line arguments for dataset and dataset hierarchy names
     if 3 == len(sys.argv):
          # Input Dataset name
          dataset_input_name = sys.argv[1]
          # print("Input dataset name : %s" %(dataset_input_name))
          # Output Dataset name
          dataset_output_name = sys.argv[2]
          # print("Output dataset name : %s" %(dataset_output_name))
          # Retrieve the input dataset pattern information from the dataset (FD-value based)
          pattern_info_i = retrieve_pattern_info(dataset_input_name, FD_INPUT_INFO)
          # Retrieve the output dataset pattern information from the dataset (FD-value based)
          pattern_info_o = retrieve_pattern_info(dataset_output_name, FD_OUTPUT_INFO)
          # Get the intersection of patterns 
          pattern_info_o = pattern_info_i.intersection(pattern_info_o)
          print("Unique patterns size :", len(pattern_info_o))
          print("Unique patterns :", pattern_info_o)
     else:
          # Incorrect command line arguments passed
          # print("Usage:")
          print("python <program_name.py> <dataset_input_name> <dataset_output_name>")
