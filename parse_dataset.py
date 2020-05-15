import re
import copy
import time
import pickle 
import numpy as np
import pandas as pd
from hierarchy_to_tree import *
#from generalization import *
from hash_generation import *
from difflib import SequenceMatcher


# Dataset Information - [{Generalization Hierarchy} + {Patterns Information} + [Clustering Information] + [QID Information]]
# Generalization Hierarchy :
# Key - "<attr_name>_<attr_val>"
# Value - [Parent, Distance]
#       - Parent of the attribute value 
#       - Distance of the parent (no of children)   
# Pattern Information :
# Key - "attr_1_val_1 attr_2_val_1 [depending on the FD]"
# Value - [Utility, [FDs]] per pattern 
#       - Pattern preserve metric information 
#       - FDs information to which the patterns belong. Ex: [0, 1]  
# Clustering Information :
# Value - [[Clustered classes], Sealed counter]
#       - Clustered class information within the clustering 
#       - Sealed cluster counter information 
# QID Information
# Pandas data frame object is appended to the dataset_info object after the dataset is parsed

DGH_INFO_IDX = 0x00
DGH_INFO_PARENT_IDX = 0x00
DGH_INFO_DISTANCE_IDX = 0x01
DGH_INFO_UNCERTAINITY_IDX = 0x02

PATTERN_INFO_IDX = 0x01
PATTERN_INFO_METRIC_START_IDX = 0x00
PATTERN_INFO_METRIC_CURRENT_IDX = 0x01
PATTERN_INFO_METRIC_BEST_IDX = 0x02
PATTERN_INFO_FDS_IDX = 0x03
PATTERN_INFO_BEST_CLASSES_START_IDX = 0x04
PATTERN_INFO_BEST_CLASSES_CURRENT_IDX = 0x05
PATTERN_INFO_BEST_CLASSES_BEST_IDX = 0x06

CLUSTERING_INFO_IDX = 0x02
CLASS_INFO_IDX = 0x00
SEALED_COUNTER_INFO_IDX = 0x01

QID_INFO_IDX = 0x03

# Global dependency generalization data store
dataset_info = [{}, {}, [], {}]

# Cache hierarchy information pertaining to the parent and distance between two nodes  
hierarchy_cache = {}
# Cache the preserve values of the patters with respect to a clustered class
patterns_preserve_cluster_cache = {}
# Cache the patterns in the classes belonging to a cluster 
cluster_patterns_cache = {} 
# Cache the sealed cluster patterns with preserve set to 1
sealed_patterns_cache = set()
# Cache the generalized classes information 
generalized_classes_cache = {}


# Optimization 
# Class distance information to generate combinations
# Sealed class index 
SEALED_CLASS_INFO_IDX = 0
# Clustered classes 
CLUSTERED_CLASS_INFO_IDX = 1
# Classe distance index 
CLASSES_DISTANCE_INFO_IDX = 2

# Threshold settings for best classes size 
BEST_CLASSES_SIZE_THRESHOLD = 10 
# Nearest classes threshold used to generate combinations 
NEAREST_CLASSES_COMBINATION_THRESHOLD = 1000



# Newyork dataset attributes  
ATT_NAMES = ['borough',                 #0
             'address',                 #1 
             'violationcode',           #2
             'violationdesc',           #3
             'zipcode',                 #4
             'cuisinedesc',             #5
             'action',                  #6
             'criticalflag',            #7
             'inspectiontype',          #8
             'score',                   #9 (sensitive) 
             'grade'                    #10
             ]          
# QID attributes (should match the hierarchy links as we use these names as part of the hierrachy keys)
QI_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
QI_ATT_NAMES = ['borough', 'address', 'violationcode', 'violationdesc', 'zipcode', 'cuisinedesc', 'action', 'criticalflag', 'inspectiontype', 'grade']
# Sensitive attribute information (maps to the QI_INDEX)
MIN_VALUE = 0.0
MAX_VALUE = 100.0
# Sensitive attribute information (maps to the QI_INDEX)
SENSITIVE_ATT_INFO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Hierarchy links (should match the QIDs as we use these names as part of the hierrachy keys) 
HIERARCHY_LINKS = ["./data/NY/borough.txt", 
                   "./data/NY/address.txt", 
                   "./data/NY/violationcode.txt", 
                   "./data/NY/violationdesc.txt",
                   "./data/NY/zipcode.txt",
                   "./data/NY/cuisinedesc.txt",
                   "./data/NY/action.txt",
                   "./data/NY/criticalflag.txt",
                   "./data/NY/inspectiontype.txt",
                   "./data/NY/grade.txt"
                   ]
# Functional dependencies (FDs)
#FD_INFO = [[1, 0], [2, 3], [4, 0], [5, 1, 0], [7, 8, 6], [7, 9, 10] ]
FD_INFO = [[1, 0], [2, 3], [4, 0], [5, 1, 0]]
#FD_INFO = [[1, 0], [4, 0], [5, 1, 0]]  #pickle distance fds




'''
# Drug dataset attributes  
ATT_NAMES = [
             'sex',                      #0 
             'race',                     #1
             'age',                      #2
             'zipcode',                  #3
             'residencestate',           #4
             'residencecounty',          #5
             'deathcity',                #6
             'deathcounty',              #7
             'location',                 #8
             'descriptionofinjury'       #9
            ]
# QID attributes (should match the hierarchy links as we use these names as part of the hierrachy keys)
QI_INDEX = [
            0, 
            1, 
            2, 
            3, 
            4, 
            5, 
            6, 
            7,
            # 8,
            # 9
           ]
QI_ATT_NAMES = [
                'sex',                   #0 
                'race',                  #1
                'age',                   #2
                'zipcode',               #3
                'residencestate',        #4
                'residencecounty',       #5
                'deathcity',             #6
                'deathcounty',           #7
                # 'location',              #8
                # 'descriptionofinjury'    #9
               ]
# Sensitive attribute information (maps to the QI_INDEX)
MIN_VALUE = 0.0
MAX_VALUE = 100.0
SENSITIVE_ATT_INFO = [
                      0, 
                      0, 
                      0, 
                      0, 
                      0, 
                      0, 
                      0, 
                      0,
                      # 0,
                      # 0
                     ]
# Hierarchy links (should match the QIDs as we use these names as part of the hierrachy keys) 
HIERARCHY_LINKS = [
                   "./data/drug/sex.txt", 
                   "./data/drug/race.txt", 
                   "./data/drug/age.txt", 
                   "./data/drug/zipcode.txt",
                   "./data/drug/residencestate.txt",
                   "./data/drug/residencecounty.txt",
                   "./data/drug/deathcity.txt",
                   "./data/drug/deathcounty.txt",
                   # "./data/drug/location.txt"
                  ]
# Functional dependencies (FDs)
FD_INFO = [
           [3, 4], 
           [5, 4], 
           [6, 7],
           # [8, 9] # Sensitive FD (8, 9)
          ]
'''



'''
# Patient dataset attributes
ATT_NAMES = [
             'age',           #0
             'postal-codes',  #1
             'state',         #2
             'diagnosis',     #3
             'medication'     #4
            ]    
# QID attributes (should match the hierarchy links as we use these names as part of the hierrachy keys)
QI_INDEX = [
            0, 
            1, 
            2, 
            3, 
            4
           ]
QI_ATT_NAMES = [
                'age',           #0
                'postal-codes',  #1
                'state',         #2
                'diagnosis',     #3
                'medication'     #4
               ]
# Sensitive attribute information (maps to the QI_INDEX)
MIN_VALUE = 10.0
MAX_VALUE = 89.0
SENSITIVE_ATT_INFO = [
                      1, 
                      0, 
                      0, 
                      0, 
                      0
                     ]
# Hierarchy links (should match the QIDs as we use these names as part of the hierrachy keys)
HIERARCHY_LINKS = [
                   "./data/patient/age.txt", 
                   "./data/patient/postal-codes.txt",
                   "./data/patient/state.txt", 
                   "./data/patient/diagnosis.txt", 
                   "./data/patient/medication.txt"
                  ]
# Functional dependencies (FDs)
# FD_INFO = [
             [0, 1],
             [1, 2], 
             [3, 4],
            ]
'''


'''
# Adult dataset attributes
ATT_NAMES = ['age',             #0
             'work-status',     #1
             'education',       #2
             'marriage-status', #3
             'family-status',   #4
             'ethnicity',       #5
             'gender',          #6
             'country',         #7
             'salary']          #8
# QID attributes (should match the hierarchy links as we use these names as part of the hierrachy keys)
QI_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8]
QI_ATT_NAMES = ['age', 'work-status', 'education', 'marriage-status', 'family-status', 'ethnicity', 'gender', 'country', 'salary']
# Sensitive attribute information (maps to the QI_INDEX)
SENSITIVE_ATT_INFO = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# Hierarchy links (should match the QIDs as we use these names as part of the hierrachy keys)
HIERARCHY_LINKS = ["./data/adult/age.txt", "./data/adult/work-status.txt", "./data/adult/education.txt", 
                   "./data/adult/marriage-status.txt", "./data/adult/family-status.txt", "./data/adult/ethnicity.txt", 
                   "./data/adult/gender.txt", "./data/adult/country.txt", "./data/adult/salary.txt"]
# Functional dependencies (FDs)
# FD_INFO = [[0, 1], [2, 3, 6, 7]]
FD_INFO = [[0, 1]]
'''



# Read dataset 
# Retrive the QID information 
# Clean the QID information 
# Store the QID information
def read_dataset(dataset_name):
     # print("########################## Read dataset")
     qid_info = {}
     match = re.search(r'data/', dataset_name)
     if match is None:
          dataset_name = "data/" + dataset_name
     dataset_qid_name = dataset_name + ".qid"
     with open(dataset_name, 'r') as in_file, open(dataset_qid_name, 'w') as out_file:
          # Read the adult dataset information 
          for num, line in enumerate(in_file, start=0):
               data = None
               # Remove any leading and trailing spaces in the string 
               strip_line = line.strip()
               # Split the string using delimiter(result into a list)
               split_line = strip_line.split(',')
               # print(num, split_line)          
               # Retrieve only the QID information 
               for indx in QI_INDEX: 
                    # Retrieve QIDs from the split list and remove any leading and trailing spaces 
                    qid = split_line[indx].strip()
                    # print(qid)
                    # Validate the QID information - skip that tuple if invalid
                    if qid is None or len(qid) == 0 or qid == "?": 
                         data = None
                         break 
                    # Append all the valid QIDs to a string 
                    if not data: 
                         data = qid
                    else:
                         data = data + "," + qid 
               # print(data)
               if data is not None:
                    # Write the QID information to a file on the disk
                    out_file.write(data + "\n")
                    # Store the QID information into a dictionary
                    qid_info[num] = data.split(",")
     # Store the QID dictionary information into the dataset information 
     dataset_info[QID_INFO_IDX] = qid_info
     print("\nQID data information:\n", qid_info)
     # print("#################################################")



# Read generalization hierarchy links -> JSON trees -> DGH object 
def read_dataset_hierarchies(hierarchy_links, dataset, attr_info):
     # print("##########################  Read dataset hierarchy links")
     # Generate the JSON trees from the hierarchy links
     json_trees_list = generate_trees(hierarchy_links)       
     # Create DGH information from the generated JSON trees
     dgh_info = create_hash(json_trees_list, dataset, attr_info)
     # Store the generated DGH object 
     dataset_info[DGH_INFO_IDX] = dgh_info
     # print("DGH info :", dgh_info)
     # print("#################################################")



# Retrieve the parent and distance information from the generalization hierarchy 
def retrieve_hierarchy_info(dgh_info, attr_info, node_1, node_2):
     retrieve_start_time = time.time()
     # print("##########################  Retrieve hierarchy information")
     # print("Attribute info : %s" %(attr_info))
     # print("Node 1 : %s" %(node_1))
     # print("Node 2 : %s" %(node_2))
     try:
          # Check the cache for the hierarchy information of the nodes  
          cache_key = attr_info + "_" + node_1 + "_" + node_2
          return hierarchy_cache[cache_key]
     except:
          parent_list_1 = []
          parent_list_2 = []
          # DGH key - <attribute-name>_node1_node2
          # Retrieve attribute root information 
          dgh_key = attr_info + "_" + "*" 
          root_data = dgh_info[dgh_key]
          # Retrieve node_1 information 
          dgh_key = attr_info + "_" + node_1 + "_" + node_1
          dgh_data = dgh_info[dgh_key]
          parent_list_1.append(dgh_data)
          # Retrieve node_2 information 
          dgh_key = attr_info + "_" + node_2 + "_" + node_2
          dgh_data = dgh_info[dgh_key]
          parent_list_2.append(dgh_data)
          # Nodes are same return the same node as parent
          if node_1 == node_2: 
               dgh_data = parent_list_1.pop()
               # Distance(Node1, Node2) = [Parent(E(v))]/[Attribute Root(E(v))]
               parent = node_1 = node_2 
               distance = float(dgh_data[DGH_INFO_UNCERTAINITY_IDX])/float(root_data[DGH_INFO_UNCERTAINITY_IDX])
               # Cache for the hierarchy information of the nodes  
               hierarchy_cache[cache_key] = (parent, float(distance))      
               # print("Nodes (%s, %s) -> Parent(%s)" %(node_1, node_2, parent))
               # print("Parent_Ev(%f)/Root_Ev(%f) - Distance (%f)" %(dgh_data[DGH_INFO_UNCERTAINITY_IDX], root_data[DGH_INFO_UNCERTAINITY_IDX], distance))
               # print("#################################################")
               retrieve_end_time = time.time()
               # print("Retrieve hierarchy execution :", retrieve_end_time - retrieve_start_time, "secs")
               return parent, float(distance) 
          # Add the attribute name to the nodes to create the DGH key 
          node_1 = attr_info + "_" + node_1 
          node_2 = attr_info + "_" + node_2 
          # Update the parent list with node_1's parent info
          dgh_data = dgh_info[node_1]
          parent_list_1.append(dgh_data)
          while True:
               # Reached root node for the attribute value 
               if dgh_data[DGH_INFO_PARENT_IDX].find("_*") != -1:
                    break
               dgh_data = dgh_info[dgh_data[DGH_INFO_PARENT_IDX]]
               parent_list_1.append(dgh_data)
          # Update the parent list with node_2's parent info
          dgh_data = dgh_info[node_2]
          parent_list_2.append(dgh_data)
          while True:
               # Reached root node for the attribute value 
               if dgh_data[DGH_INFO_PARENT_IDX].find("_*") != -1:
                    break
               dgh_data = dgh_info[dgh_data[DGH_INFO_PARENT_IDX]]
               parent_list_2.append(dgh_data)
          # print(parent_list_1) 
          # print(parent_list_2)
          # Get the smallest parent list size
          parent_list_size = len(parent_list_1)
          if parent_list_size > len(parent_list_2): 
               parent_list_size = len(parent_list_2) 
          # Find the common parent of the attributes 
          prev_parent_node = None
          while parent_list_size:  
               curr_parent_node_1 = parent_list_1.pop()
               curr_parent_node_2 = parent_list_2.pop()
               # Current parent nodes are not same return parent at one level higher
               if curr_parent_node_1 != curr_parent_node_2:
                    break 
               # Store the previous parent node 
               prev_parent_node =  curr_parent_node_1 =  curr_parent_node_2
               parent_list_size -= 1
          # Retrieve the parent uncertainity value (Root(Ev))
          split_list = prev_parent_node[DGH_INFO_PARENT_IDX].split("_")
          parent = split_list[1] 
          dgh_key = attr_info + "_" + parent + "_" + parent  
          parent_data = dgh_info[dgh_key]
          # Distance(Node1, Node2) = [Parent(E(v))]/[Attribute Root(E(v))]
          distance = float(parent_data[DGH_INFO_UNCERTAINITY_IDX])/float(root_data[DGH_INFO_UNCERTAINITY_IDX])
          # Cache for the hierarchy information of the nodes  
          hierarchy_cache[cache_key] = (parent, float(distance))      
          # print("Nodes (%s, %s) -> Parent(%s)" %(node_1, node_2, parent))
          # print("Parent_Ev(%f)/Root_Ev(%f) - Distance (%f)" %(parent_data[DGH_INFO_UNCERTAINITY_IDX], root_data[DGH_INFO_UNCERTAINITY_IDX], distance))
          retrieve_end_time = time.time()
          # print("Retrieve hierarchy execution :", retrieve_end_time - retrieve_start_time, "secs")
          return parent, distance 
          # print("#################################################")
  


def calculate_sensitive_attribute_distance(attr_info, value_1, value_2): 
     sensitive_start_time = time.time()
     # print("##########################  Calculate sensitive attribute distance")
     # print("Value 1 : %s" %(value_1))
     # print("Value 2 : %s" %(value_2))
     try:
          # Check the cache for the sensitive attribute distance 
          cache_key = attr_info + "_" + value_1 + "_" + value_2
          return hierarchy_cache[cache_key]
     except:
          parent = 'None'
          try:
               value_1 = float(value_1)
               value_2 = float(value_2)
               # Values are of numerical type - calculate range
               distance = abs(value_1 - value_2)/float(MAX_VALUE - MIN_VALUE)
          except:  
               # Values are of string type - calculate string distance (identical strings returns 1.0)
               distance = 1.0 - (SequenceMatcher(None, value_1, value_2).ratio())
          # Cache for the sensitive information of the nodes  
          hierarchy_cache[cache_key] = (parent, float(distance))      
          # print("Sensitive values (%s, %s) - distance (%f)\n" %(value_1, value_2, distance))
          sensitive_end_time = time.time()
          # print("Sensitive attribute distance execution :", sensitive_end_time - sensitive_start_time, "secs")
     return parent, distance 
     # print("#################################################")



# Retrieve the pattern information from the dataset (FD-value based)
def retrieve_pattern_info(dataset_name, k_size):
     # print("##########################  Retrieve pattern info")
     match = re.search(r'data/', dataset_name)
     if match is None:
          dataset_name = "data/" + dataset_name
     # Open the dataset in read only mode
     with open(dataset_name, 'r') as in_file:
          # Retrive the pattern inforamtion 
          pattern_info = dataset_info[PATTERN_INFO_IDX] 
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
                    try: 
                         # Retrieve the pattern information associated with the key
                         pattern_data = pattern_info[pattern_key]
                    except: 
                         # Empty list of lists to store FDs and best classes to which the pattern belongs 
                         pattern_init = [0, 0, 0, [], set(), set(), set()]
                         # Pattern information desn't exist - create one 
                         pattern_info[pattern_key] = pattern_init    
                         # Retrieve the pattern information associated with the key
                         pattern_data = pattern_info[pattern_key]
                    # Store the FD information pertaining to the pattern 
                    pattern_data[PATTERN_INFO_FDS_IDX] = item  
                    # Calculate the default preserve metric value 
                    # Preserve = (1 - min(Distance)) * C/K = (1 - 0) * 1/K
                    pattern_data[PATTERN_INFO_METRIC_START_IDX] = 1/float(k_size)
                    pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX] = 1/float(k_size)
                    pattern_data[PATTERN_INFO_METRIC_BEST_IDX] = 1/float(k_size)
                    # Best classes for that particular pattern generating the default preserve value 
                    # Reason for adding this threshold value is to avoid unecessary performance overhead storing too many values
                    if (len(pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX]) < BEST_CLASSES_SIZE_THRESHOLD):
                         pattern_data[PATTERN_INFO_BEST_CLASSES_START_IDX].add(num) 
                         pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX].add(num)
                         pattern_data[PATTERN_INFO_BEST_CLASSES_BEST_IDX].add(num)
                    # print("key :", pattern_key, "Value :", pattern_data)
          # Store the updated pattern information 
          dataset_info[PATTERN_INFO_IDX] = pattern_info
     print("Pattern info :", pattern_info)
     print("Pattern info count :", len(pattern_info.keys()))
     # print("#################################################")



# Generate clustering information from the QID inforamtion(singleton classes)
def generate_clustering_info(dataset_qid_name):
     # print("##########################  Generate clustering info")
     match = re.search(r'data/', dataset_qid_name)
     if match is None:
          dataset_qid_name = "data/" + dataset_qid_name
     # Open the dataset in read only mode
     with open(dataset_qid_name, 'r') as in_file:
          # Class ID information 
          class_ids = [] 
          # Read the dataset information one line at a time  
          for num, line in enumerate(in_file, start=0):
               # Generate the singleton class information 
               class_ids.append(num)
          # Store the generated class information 
          clustering_info = dataset_info[CLUSTERING_INFO_IDX]
          clustering_info.insert(CLASS_INFO_IDX ,class_ids)   
          clustering_info.insert(SEALED_COUNTER_INFO_IDX, 0)
     print("Clustering info :", clustering_info)
     # print("#################################################")



# Sort the QID data frame information  
def sort_dataset():
     # print("##########################  Sort dataset")
     # Retrieve the QID data frame information 
     qid_data_frame = dataset_info[QID_INFO_IDX]
     # Sort data frames based on the coloumn index (ascending by default)
     qid_data_frame = qid_data_frame.sort_values(QI_ATT_NAMES) 
     # print(qid_data_frame)
     # print("#################################################")



# Get the parsed dataset inforamtion
def get_parsed_dataset_info(dataset_name, k_size):
     # print("##########################  Get parsed dataset info")
     # Read the dataset and generate the QID information
     read_dataset(dataset_name)
     # Retrieve the pattern information
     dataset_qid_name = dataset_name + ".qid"
     retrieve_pattern_info(dataset_qid_name, k_size)
     # Generate clustering information 
     generate_clustering_info(dataset_qid_name)
     # Parse the hierarchy information 
     read_dataset_hierarchies(HIERARCHY_LINKS, dataset_name, ATT_NAMES)
     # print("#################################################")
     return dataset_info



# Calculate the Clustering Generalization Utility(CGU) metric information   
def calculate_clustering_generalization_utility(pattern_info):
     cgu_start_time = time.time()
     # print("##########################  Calculate clustering generalization Utility")
     preserve = 0.0 
     # print("Pattern info : ", pattern_info)
     # Retrieve all the items in the pattern information  
     for key, value in pattern_info.items():
          # Retrieve the stored preserve for the pattern
          preserve += value[PATTERN_INFO_METRIC_CURRENT_IDX]
     # CGU = CGP of every pattern/Number of patterns      
     utility = float(preserve) / len(pattern_info)
     # print("Total patterns(%d) preserve : (%f)" %(len(pattern_info), preserve))
     # print("CGU :", utility)
     cgu_end_time = time.time()
     # print("CGU execution :", cgu_end_time - cgu_start_time, "secs")
     # print("#################################################")
     return utility 



# Calculate the preserve of single pattern with respect to a clustered class  
def calculate_single_pattern_preserve_cluster(_class, pattern, pattern_info, qid_info, k_size):
     pattern_start_time = time.time()
     # print("##########################  Calculate single pattern preserve - cluster")
     min_distance = -1.0 
     preserve = 0.0
     sensitive_present = False 
     cluster = [] 
     # Retrieve the DGH information  
     dgh_info = dataset_info[DGH_INFO_IDX]
     # Split the pattern to attribute values 
     pattern_attr_list = pattern.split()
     # Retrieve the pattern data associated with the pattern 
     pattern_data = pattern_info[pattern] 
     # print("Pattern data :", pattern_data)
     # Retrieve the FD information from the pattern data 
     fd_data = pattern_data[PATTERN_INFO_FDS_IDX]
     # print("Pattern FD :", fd_data)
     # Validate the class size 
     if type(_class) is list:
          # Add the clustered class information
          for item in _class:
               cluster.append(item)
     else:
          # Add the singleton class information 
          cluster.append(_class)
     # Cluster size 
     c_size = len(cluster)
     # Any sensitive attributes 
     for attr in fd_data:
          if SENSITIVE_ATT_INFO[attr]:
               sensitive_present = True
               break  
     # Loop through all the classes within the cluster 
     for _class in cluster: 
          attr_list_cnt = 0
          curr_distance = 0.0
          # Retrieve the QID informtion  
          class_info = qid_info[_class]
          # print("Class info [%u] : " %(_class), class_info)     
          # Loop through the FD data for the pattern
          for fd_indx in fd_data:
               # Retrieve the attribute value based on the FD 
               attr_value = class_info[fd_indx]
               # Calculate the distance between attribute values and pattern information 
               value_1 = pattern_attr_list[attr_list_cnt]
               value_2 = attr_value
               # QID information with sensitive attribute information 
               if SENSITIVE_ATT_INFO[fd_indx]:
                    attr_parent, attr_distance = calculate_sensitive_attribute_distance(QI_ATT_NAMES[fd_indx], value_1, value_2) 
               else:
                    # QID information with insensitive attribute information(generalized)
                    attr_parent, attr_distance = retrieve_hierarchy_info(dgh_info, QI_ATT_NAMES[fd_indx], value_1, value_2)
               # print("(Pattern value (%s), Attribute value (%s) - Distance(%d))" %(value_1, value_2, attr_distance))
               curr_distance += attr_distance
               attr_list_cnt += 1
          # Distance([J5B, QC], [JKL, BC]) = (Distance([J5B, JKL]) + Distance([QC, BC]))/Number of attributes in pattern 
          curr_distance = curr_distance / float(len(pattern_attr_list))
          # print("Pattern attributes distance : ", curr_distance)
          # Store the minimal total distance 
          if min_distance < 0 or min_distance > curr_distance:
               min_distance = curr_distance
          # Sensitive attributes are not present then break from the loop
          # QIDs without sensitive are generalised so calculating the distance only once would suffice 
          if sensitive_present == False:
               break
     # Calculate the preserve metric for the pattern 
     # print("Minimal total distance (%s) : %f" %(pattern, min_distance))
     preserve = (1 - min_distance) * (c_size / float(k_size))
     # print("Preserve (%s) : %f" %(pattern, preserve))
     pattern_end_time = time.time()
     # print("Single pattern preserve execution :", pattern_end_time - pattern_start_time, "secs")
     # print("#################################################")
     return preserve



# Calculate the preserve for single pattern with respect to a clustering  
def calculate_single_pattern_preserve_clustering(clustering, pattern, pattern_info, qid_info, k_size):
     patterns_start_time = time.time()
     # print("##########################  Calculate single pattern preserve - clustering")
     # print("k_size :", k_size)
     # print("Pattern :", pattern)
     curr_preserve = max_preserve = 0.0
     best_classes = set() 
     # Loop through all the classes in the clustering
     for _class in clustering[CLASS_INFO_IDX]:
          # Calculate the preserve of single pattern with respect to a clustered class  
          curr_preserve = calculate_single_pattern_preserve_cluster(_class, pattern, pattern_info, qid_info, k_size)
          # Replace the best classes information and update the pattern preserve to the maximum value
          if curr_preserve > max_preserve:
               # print("Replace best classes :", best_classes, "->", (_class))
               # print("Update the max preserve (%f -> %f)" %(max_preserve, curr_preserve))
               best_classes.clear() 
               max_preserve = curr_preserve
               try:
                    best_classes.add(tuple(_class))
               except:
                    best_classes.add(_class)
          # Add the class information to  best classes information
          elif curr_preserve == max_preserve:
               # Set threshold to indicate the number of entries we can add (performance optimization)
               if len(best_classes) < BEST_CLASSES_SIZE_THRESHOLD: 
                    # print("Add to best classes :", (_class), "->", best_classes)
                    try:
                         best_classes.add(tuple(_class))
                    except:
                         best_classes.add(_class)
     # print("Pattern (%s) - Max preserve (%f), " %(pattern, max_preserve))
     # print("Pattern (%s) - Best classes :" %(pattern), best_classes)
     patterns_end_time = time.time()
     # print("Single pattern preseve(clustering) execution :", patterns_end_time - patterns_start_time, "secs")
     # print("#################################################")
     return max_preserve, best_classes



def calculate_patterns_preserve_cluster(clustering, cluster_info, pattern_info, changed_patterns, greater_preserve_patterns, qid_info, k_size):
     patterns_start_time = time.time()
     # print("##########################  Calculate all patterns preserve - cluster")
     # print("k_size :", k_size)
     # print("Cluster info :", cluster_info)
     # print("Pattern info(before) :", pattern_info)
     cluster_patterns = set() 
     try:
        cluster_patterns = cluster_patterns_cache[str(cluster_info)]
     except: 
        original_qid_info = dataset_info[QID_INFO_IDX]
        # Generate the patterns within the cluster
        for _class in cluster_info: 
             qid_data = original_qid_info[_class]
             # Iterate through the FDs to generate keys to store pattern information 
             for item in FD_INFO:
                  pattern_key = None  
                  for i in item:
                       if pattern_key is None:
                            pattern_key = qid_data[i] 
                       else: 
                            pattern_key = pattern_key + " " + qid_data[i]
                  # Add the generated pattern to the set
                  cluster_patterns.add(pattern_key)     
        # Add to the cluster pattern cache 
        cluster_patterns_cache[str(cluster_info)] = cluster_patterns
     # print("Cluster patterns :", cluster_patterns)
     # print("Sealed patterns cache:", sealed_patterns_cache)
     # Ensure that the patterns that we calculate the preserve value are not part of the sealed patterns cache
     cluster_patterns = cluster_patterns - sealed_patterns_cache
     # print("Cluster patterns - Sealed patterns :", cluster_patterns)
     # Loop through only the patterns in the cluster 
     for pattern in cluster_patterns:
          curr_preserve = 0.0
          cache_key = str(cluster_info) + "_" + pattern
          # print("------> Pattern(start) : (%s) <-------" %(pattern))
          try:
               # Check the cache to retrieve the preserved value for a pattern with respect to a preserved class
               curr_preserve = patterns_preserve_cluster_cache[cache_key]  
          except:
               # Calculate single pattern preserve with respect to a clustered class 
               curr_preserve = calculate_single_pattern_preserve_cluster(cluster_info, pattern, pattern_info, qid_info, k_size)
               # Cache the calculated preserve value for the pattern with respect to the clustered class 
               patterns_preserve_cluster_cache[cache_key] = curr_preserve 
          # Retrieve the pattern data associated with the pattern 
          pattern_data = pattern_info[pattern] 
          # Retrieve the stored pattern preserve 
          stored_preserve = pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX]   
          # Retrieve the stored best classes for the pattern
          stored_classes = pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX]
          # Calculated preserve value for the pattern is better compared to the stored preserve value 
          # Replace the best class information and also update the pattern preserve value     
          # Example : P1 - ([(1,2), (3,4)], 0.5)
          #           Calculated Preseve [5,6] = 0.8; 0.8 > 0.5 so replace P1 - ([5,6], 0.8)
          if (curr_preserve > stored_preserve): 
               # print("Pattern (%s) - Preserve greater(%f > %f)" %(pattern, curr_preserve, stored_preserve))
               pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX] = curr_preserve
               # print("------> Replacing the pattern (%s) classes" %(pattern), "(", stored_classes, "->", [tuple(cluster_info)], ")")
               pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX] = set()
               pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX].add(tuple(cluster_info))
               # Update the changed patterns set 
               changed_patterns.add(pattern)
               # Update the greater preserve patterns 
               greater_preserve_patterns[pattern] = [curr_preserve, stored_preserve]               
          # Calculated preserve value for the pattern is equal to the stored preserve value 
          # Add to the best class information and no update to the pattern preserve value  
          # Example : P1 - ([(1,2), (3,4)], 0.5)
          #           Calculated Preseve [5,6] = 0.5; 0.5 = 0.5 so add to P1 - ([(1,2), (3,4), (5,6)], 0.5)
          elif (curr_preserve == stored_preserve):
               # For now, setting the number of values that we can store to 5 which can be changed    
               # Reason for adding this threshold value is to avoid unecessary performance overhead storing too many values
               if (len(pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX]) < BEST_CLASSES_SIZE_THRESHOLD):
                    # print("Pattern (%s) - Preserve equal : %f" %(pattern, curr_preserve))
                    # print("------> Adding to the pattern (%s) classes" %(pattern), "(", [tuple(cluster_info)], "->", stored_classes, ")")
                    # Shallow copy the current best classes set (don't make changes directly will reflect the start best classes set) 
                    current_set_copy = set(pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX])
                    current_set_copy.add(tuple(cluster_info))
                    pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX] = current_set_copy
                    # Update the changed patterns set 
                    changed_patterns.add(pattern)
          # Calculated preserve value for the pattern is not better compared to the stored preserve value
          # Case 1: Check if the classes with the cluster are part of the best classes for the pattern. If present, remove 
          #         those classes from the best classes else no change is required. 
          # Case 2: After removing the classes from the best classes for the pattern and it becomes empty then recompute. 
          # Example : P1 - ([(1,2), (3,4), (5,6)], 0.5)
          #           Calculated Preseve [4,5] = 0.3; 0.3 < 0.5 so update P1 - ((1,2), 0.5)
          # Example : P1 - ([(1,2], (3,4], (5,6)], 0.5)
          #           Calculated Preseve [7,8] = 0.3; 0.3 < 0.5 no update P1 - ([(1,2), (3,4), (5,6)], 0.5)
          # Example : P1 - ([(1,2], (3,4), (5,6)], 0.5)
          #           Calculated Preseve [1,2] = 0.3; 0.3 < 0.5 so update P1 - ([(3,4), (5,6)], 0.5)
          # Example : P1 - ((1,2,3,4), 0.5)
          #           Calculated Preseve [1,2] = 0.3; 0.3 < 0.5 so update P1 - (Empty) 
          #           Re-calculate the best classes and preserve value 
          else:
               # print("Pattern (%s) - Preserve lesser(%f < %f)" %(pattern, curr_preserve, stored_preserve))
               removal_classes = set()
               for i in cluster_info: # Always individual elements in the clustered class  Ex: [1,2] - Bound by K
                    for j in stored_classes: # Can be a tuple or element in the best classes Ex: ((1,2), (3,4), 5)
                        if type(j) is tuple and i in j: # Tuple 
                             removal_classes.add(j)
                        elif type(j) is not tuple: # Value 
                             if i == j:
                                  removal_classes.add(j)
               # Get the final updated classes after removing the classes that led to lower preserve value 
               final_classes = stored_classes - removal_classes  
               # Final computed best classes is same as stored best classes 
               if (final_classes == stored_classes):
                    # Do nothing 
                    pass
                    # print("------> No change required")
               # Final computed best classes set is not empty 
               elif len(final_classes):
                    # print("------> Updating the pattern (%s) classes" %(pattern), "(", stored_classes, "->", (final_classes), ")")
                    pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX] = final_classes
                    # Update the changed patterns set 
                    changed_patterns.add(pattern)
               else: 
                    ''' Computationally expensive and degradation in performance due to recomputing the best preserve over all classes when empty 
                    # Re-compute the pattern preserve value as best classes are empty
                    new_preserve, best_classes = calculate_single_pattern_preserve_clustering(clustering, pattern, pattern_info, qid_info, k_size)
                    # print("------> Recompute the pattern (%s) preserve (%f -> %f)" %(pattern, stored_preserve, curr_preserve))
                    pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX] = new_preserve 
                    # print("------> Recompute the pattern (%s) classes" %(pattern), "(", stored_classes, "->", best_classes, ")")
                    pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX] = best_classes 
                    # Update the changed patterns set 
                    changed_patterns.add(pattern)
                    ''' 
                    # Instead of the above recomputation calculate the default preserve metric value and assign the same
                    # Preserve = (1 - min(Distance)) * C/K = (1 - 0) * 1/K
                    pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX] = 1/float(k_size) # Class Size(C) = 1
                    # Reset the best classes (would get recomputed accordingly later)
                    pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX] = set()
                    # Update the changed patterns set 
                    changed_patterns.add(pattern)
     # print("Pattern info(after) : ", pattern_info)
     patterns_end_time = time.time()
     # print("Patterns preseve(cluster) execution :", patterns_end_time - patterns_start_time, "secs")
     # print("#################################################")



# Calculate the preserve for all the patterns with respect to a clustering
def calculate_patterns_preserve_clustering(clustering, pattern_info, changed_patterns, greater_preserve_patterns, qid_info, k_size):
     patterns_start_time = time.time()
     # print("##########################  Calculate all patterns preserve - clustering")
     # print("k_size :", k_size)
     # print("Pattern info(before) :", pattern_info)
     # Loop through the patterns in the pattern list
     for pattern in pattern_info.keys():
          curr_preserve = 0.0
          # print("------> Pattern(start) : (%s) <-------" %(pattern))
          # Loop through all the classes in the clustering
          for _class in clustering[CLASS_INFO_IDX]:
               # Calculate the preserve of single pattern with respect to a clustered class
               curr_preserve = calculate_single_pattern_preserve_cluster(_class, pattern, pattern_info, qid_info, k_size)
               # Retrieve the pattern data associated with the pattern
               pattern_data = pattern_info[pattern]
               # Retrieve the stored pattern preserve
               stored_preserve = pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX]
               # Update the pattern preserve to the maximum value
               if curr_preserve > stored_preserve:
                    # print("------> Updating the pattern (%s) preserve to %f" %(pattern, curr_preserve))
                    pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX] = curr_preserve
                    # Update the greater preserve patterns 
                    greater_preserve_patterns[pattern] = [curr_preserve, stored_preserve]               
                    # Update the changed patterns set 
                    changed_patterns.add(pattern)
     # print("Pattern info(after) : ", pattern_info)
     patterns_end_time = time.time()
     # print("Patterns preseve(clustering) execution :", patterns_end_time - patterns_start_time, "secs")
     # print("#################################################")



# Calculate the nearest neighbors with respect to a pivot class in the cluster based on the DGH 
def calculate_nearest_neighbors(cluster, qid_info):
     neighbors_start_time = time.time()
     # print("##########################  Calculate nearest neighbors")
     # print("Cluster :", cluster)
     # print("QID info :", qid_info)
     pivot_class = None
     fd_data = []
     nearest_neighbors = []
     neighbor_distance = {} 
     # Retrieve the DGH information  
     dgh_info = dataset_info[DGH_INFO_IDX]
     # Flatten the FD information into a list of indices 
     for _item in FD_INFO: 
          if type(_item) is list: 
               for _indx in _item:
                    fd_data.append(_indx)
     for _class in cluster: 
          distance = 0.0
          # Pick any class from the list of cluster classes as a pivot and compute
          # the distance with other classes
          if pivot_class == None:
               pivot_class = qid_info[_class]
               # print("Pivot class info [%u] : " %(_class), pivot_class)     
               # Set the distance of the pivot class to itself to zero 
               # Store the pivot class distance in the neighbor distance list 
               neighbor_distance[_class] = 0
               continue
          neighbor_info = qid_info[_class]
          # print("Neighbor class info [%u] : " %(_class), neighbor_info)     
          # Loop through the FD data for the class 
          for fd_indx in fd_data:
               # Calculate the distance between attribute values of the pivot and neighbor classes 
               value_1 = pivot_class[fd_indx]
               value_2 = neighbor_info[fd_indx]
               # QID information with sensitive attribute information 
               if SENSITIVE_ATT_INFO[fd_indx]:
                    attr_parent, attr_distance = calculate_sensitive_attribute_distance(QI_ATT_NAMES[fd_indx], value_1, value_2) 
               else:
                    # QID information with insensitive attribute information(generalized)
                    attr_parent, attr_distance = retrieve_hierarchy_info(dgh_info, QI_ATT_NAMES[fd_indx], value_1, value_2)
               # print("(Pivot attribute value (%s), Neighbor class attribute value (%s) - Distance(%d))" %(value_1, value_2, attr_distance))
               distance += attr_distance
          # print("Pivot class and neighbor class(%u) total distance :" %(_class), distance)
          # Store the calculated distance in the dictionary
          neighbor_distance[_class] = distance
     # Sort the distances stored in the dictionary
     sorted_by_value = sorted(neighbor_distance.items(), key=lambda kv: kv[1]) 
     # Retrieve the class information from the sorted distance list 
     for _neighbor, distance in sorted_by_value:
          nearest_neighbors.append(_neighbor)
     # print("Nearest neighbors :", nearest_neighbors)
     neighbors_end_time = time.time()
     # print("Nearest neighbors metric execution :", neighbors_end_time - neighbors_start_time, "secs")
     # print("#################################################")
     return nearest_neighbors 



# Generalize the cluster information based on the DGH 
def generalize_cluster_info(cluster_idx, qid_info):
     generalize_start_time = time.time()
     # print("##########################  Generalize cluster info")
     # print("Cluster info :", cluster_idx)
     # Retrieve the QID data frame information 
     # print("\nGeneralize cluster information:\n")
     attr_parent_info = []
     attr_cnt = 0
     try:
          # Retrieve the cached generalized cluster information 
          attr_parent_info = generalized_classes_cache[str(cluster_idx)]
          attr_cnt = len(attr_parent_info)
     except:
          cluster = []
          # Generalize the information in the generated cluster using DGH at attribute level
          dgh_info = dataset_info[DGH_INFO_IDX] 
          # Iterate through all the classes in the cluster
          for _class in cluster_idx:
               # Convert the data frame row to a list 
               class_info = qid_info[_class]
               # print("Class info [%u] : " %(_class), class_info)
               # Generate a cluster with the actual QID information    
               cluster.append(class_info)
          # print(cluster)
          cluster_cnt = len(cluster)
          # Iterate through the cluster with the QID information to find the command attribute parent 
          # Cluster is assumed to be atleast K=2 to generalize
          attr_parent_info = copy.deepcopy(cluster[0])
          attr_cnt = len(attr_parent_info)
          for _class_id in range(cluster_cnt):
               for _attr_id in range(attr_cnt):
                    value_1 = attr_parent_info[_attr_id]
                    value_2 = cluster[_class_id][_attr_id]
                    if SENSITIVE_ATT_INFO[_attr_id]:
                         # Hierarchy doesn't exist for sensitive attributes 
                         parent = 'None' 
                    else:
                         # Retrieve the common parent information for the attribute values 
                         parent, distance = retrieve_hierarchy_info(dgh_info, QI_ATT_NAMES[_attr_id], value_1, value_2)
                    # print("<----------(Attribute value 1(%s), Attribute value 2(%s) - Parent(%s), Distance(%s)" %(value_1, value_2, parent, distance))
                    # Update the attribute parent information accordingly
                    attr_parent_info[_attr_id] = parent
          # Cache the generalized cluster information
          generalized_classes_cache[str(cluster_idx)] = attr_parent_info          
     # Update the QID info with the generalized information 
     for _class in cluster_idx:
          # Retrieve the class information from the QID table that needs to be updated 
          class_info = qid_info[_class]   
          # Update the attribute information with the generalized information
          for _attr_id in range(attr_cnt):
               # Don't generalize the sensitive attribute information in the QID info dataframe
               if SENSITIVE_ATT_INFO[_attr_id]:
                    continue
               class_info[_attr_id] = attr_parent_info[_attr_id]     
     # print(qid_info)              
     generalize_end_time = time.time()
     # print("Generalize metric execution :", generalize_end_time - generalize_start_time, "secs")
     # print("#################################################")
 


# Calculate the distance of the closest clases based on the FD for each class(used to optmize the combinations)
def calculate_classes_distance(dataset_name, qid_info):
     distance_start_time = time.time()
     # print("##########################  Calculate classes distance")
     # print("Dataset :", dataset_name)
     # print("QID info :", qid_info)
     # print("QID info classes :", qid_info.shape[0])
     classes_dist = {}
     distances_file = (dataset_name.split('.csv'))[0] + '_distances.pickle'
     try:
          # Check if the distance information is already pickled 
          distances_fd = open(distances_file, 'rb')
          classes_dist = pickle.load(distances_fd)  
          print("Classes distance found in the pickled file -", distances_file)
          distances_fd.close()
          # print("Classes distance :", classes_dist)           
          distance_end_time = time.time()
          # print("Classes distance execution :", distance_end_time - distance_start_time, "secs")
          # print("#################################################")
          return classes_dist
     except:
          print("Classes distance not found in the pickled file - calculating...")
          # Generalize the information in the generated cluster using DGH at attribute level
          dgh_info = dataset_info[DGH_INFO_IDX] 
          # Calculate the distance for all the classes in the QID info based on the FDs  
          for _class_i in qid_info.keys(): 
               print("Computing class distance - Class [%d]" %(_class_i))
               # Class distance list w.r.t to compared class
               class_dist = [0, _class_i, []]          
               # Number of FDs configured 
               fd_size = len(FD_INFO)
               # Loop through all the classes to calcualte the distance
               for _class_j in qid_info.keys(): 
                    # Rest the total distance after each class comparison
                    tot_distance = 0.0
                    # Skip the distance calculation to the same class
                    if _class_i == _class_j:
                         continue
                    # FD information for the dataset 
                    for fd_cntr, _FD in enumerate(FD_INFO):
                         for _item in _FD:
                              value_1 = qid_info[_class_i][_item]
                              value_2 = qid_info[_class_j][_item]
                              # Retrieve the distance information from the DGH information
                              if SENSITIVE_ATT_INFO[_item]:
                                   # Sensitive attributes have no hierarchy 
                                   parent, distance = calculate_sensitive_attribute_distance(QI_ATT_NAMES[_item], value_1, value_2) 
                              else: 
                                   parent, distance = retrieve_hierarchy_info(dgh_info, QI_ATT_NAMES[_item], value_1, value_2)
                              # print("(%s, %s) - Parent(%s), Distance(%f)" %(value_1, value_2, parent, distance))
                              tot_distance += distance
                    # print("Total distance :", tot_distance)
                    # Store the calculated distance for the FD 
                    class_dist[CLASSES_DISTANCE_INFO_IDX].append((_class_j, tot_distance))  
               # Sort the stored distances for that particular class  
               class_dist[CLASSES_DISTANCE_INFO_IDX] = sorted(class_dist[CLASSES_DISTANCE_INFO_IDX], key=lambda tup: tup[1]) 
               # Nearest classes threshold used to generate combinations 
               class_dist[CLASSES_DISTANCE_INFO_IDX] = class_dist[CLASSES_DISTANCE_INFO_IDX][:NEAREST_CLASSES_COMBINATION_THRESHOLD]
               # Store the calculated distances for all the FDs 
               classes_dist[_class_i] = class_dist
               # print("Classes distance :", class_dist)           
          # Cache the calculated classes distances information 
          distances_fd = open(distances_file, 'wb')
          pickle.dump(classes_dist, distances_fd)  
          distances_fd.close()          
          distance_end_time = time.time()
          print("Classes distance execution :", distance_end_time - distance_start_time, "secs")
          # print("#################################################")
          return classes_dist
 
     

if __name__ == '__main__':
     # Read the dataset and generate the QID information
     read_dataset("./data/patient/patient.csv")
     # Retrieve the pattern information
     retrieve_pattern_info("./data/patient/patient.csv.qid")
     # Generate clustering information
     generate_clustering_info("./data/patient/patient.csv.qid")
