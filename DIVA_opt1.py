import copy
import csv
import re
import sys
import time
from itertools import combinations
from parse_dataset import *


# Maximum CGU threshold percentage (scale 0 - 1)
MAX_DEPENDENCY_GEN_CGU_THRESHOLD = 0.6 

# Cache the combinations information
# Updated pattern information
# Current CGU information
GREATER_PRESERVE_PATTERNS = 0x00
GREATER_PRESERVE_PATTERNS_CURR_PRESERVE = 0x00
GREATER_PRESERVE_PATTERNS_PREV_PRESERVE = 0x01
COMBINATION_CURR_CGU_INFO = 0x01
combinations_cache = {}


# Store clustered and generalized QID information  
def store_clustered_qid_info(dataset_name, clustering_info, pattern_info, qid_info, k_size):
     print("########################## Store clustered QID info")
     print("Dataset name :", dataset_name)
     print("Clustering :", clustering_info)
     print("K-size :", k_size)
     print("QID info :")
     print(qid_info)
     updated_clustering = []
     updated_clustering_info = []
     updated_qid_info = {} 
     match = re.search(r'data/', dataset_name)
     if match is None:
          dataset_name = "data/" + dataset_name
     dataset_qid_name = dataset_name + ".qid.clustered"
     for cluster in clustering_info[CLASS_INFO_IDX]:
          # Skip or suppress the classes or clusters less than K-size
          if (type(cluster) is not list) or (len(cluster) < k_size):
               continue 
          # Update the clustering (excluding the suppressed or skipped clusters) 
          updated_clustering.append(cluster)
          # Update the QID information(excluding the suppressed or skipped clusters) 
          for _class in cluster: 
               updated_qid_info[_class] = qid_info[_class]
     updated_clustering_info.insert(CLASS_INFO_IDX, updated_clustering)
     updated_clustering_info.insert(SEALED_COUNTER_INFO_IDX, 0)
     print("Updated QID info(suppresion):")
     for _clustered_class in updated_clustering_info[CLASS_INFO_IDX]:
          for _class in _clustered_class: 
               print(_class, "->", updated_qid_info[_class])
          print("<--------- ############# ------------->")
     print("Updated clustering info(suppression):")
     print(updated_clustering_info)
     # Write the QID information to a file 
     pd.DataFrame.from_dict(data=updated_qid_info, orient='index').to_csv(dataset_qid_name, header=False)
     # Reset the patterns current preserve information 
     for _key in pattern_info.keys():
          pattern_data = pattern_info[_key]
          pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX] = 0 
     # Re-calculate patterns preserve with respect to a clustering (after suppressing classes less than K-size)
     # Just the current preserve value gets updated 
     calculate_patterns_preserve_clustering(updated_clustering_info, pattern_info, set(), {}, qid_info, k_size)
     print("Updated pattern(after tuple suppression):")
     print(pattern_info)
     # Calculate the Clustering Generalization Utility(CGU) metric based on overall pattern information 
     curr_cgu = calculate_clustering_generalization_utility(pattern_info) 
     print("\nOptimal CGU : %f\n" %(curr_cgu))
     print("Pattern preserve : %f" %(len(pattern_info)))
     print("D-Loss value : %f" %(len(pattern_info) - (curr_cgu * len(pattern_info))))
     print("#################################################")

   
 
# Store clustering information based on the combination information (to revert later) 
def store_clustering_info(value_1, value_2, clustering_info):
     store_start_time = time.time() 
     # print("##########################  Store clustering information")
     # print("Value 1 :", value_1)
     # print("Value 2 :", value_2) 
     # print("Clustering info :", clustering_info)
     class_size = 0 
     class_info = {}
     indx_i = clustering_info[CLASS_INFO_IDX].index(value_1)
     indx_j = clustering_info[CLASS_INFO_IDX].index(value_2)
     if type(clustering_info[CLASS_INFO_IDX][indx_i]) is list:
          class_size += len(clustering_info[CLASS_INFO_IDX][indx_i])
          class_info[indx_i] = copy.deepcopy(clustering_info[CLASS_INFO_IDX][indx_i])  
     else: 
          class_size += 1
          class_info[indx_i] = clustering_info[CLASS_INFO_IDX][indx_i]
     if type(clustering_info[CLASS_INFO_IDX][indx_j]) is list:
          class_size += len(clustering_info[CLASS_INFO_IDX][indx_j])
          class_info[indx_j] = copy.deepcopy(clustering_info[CLASS_INFO_IDX][indx_j])  
     else: 
          class_size += 1
          class_info[indx_j] = clustering_info[CLASS_INFO_IDX][indx_j]  
     # print("Class size :", class_size)
     # print("Class info :", class_info)
     store_end_time = time.time() 
     # print("Store clustering execution time :", store_end_time - store_start_time, "secs\n")
     # print("#################################################")
     return class_info, class_size



# Generate clustering information based on the combination information 
def generate_clustering_info(value_1, value_2, clustering_info, qid_info, curr_qid_cluster, k_size):
     generate_start_time = time.time() 
     sealed_cntr = 0
     # print("##########################  Generate clustering information")
     cluster = [] 
     # print("Value 1 :", value_1)
     # print("Value 2 :", value_2) 
     # print("k_size  :", k_size)
     # print("Clustering info(before) :", clustering_info)
     # Retrieve the index information from the clustering for value 1 
     index_1 = clustering_info[CLASS_INFO_IDX].index(value_1)
     if type(value_1) is list:
          # Copy the values to the new cluster 
          for item in value_1:
              cluster.append(item)  
          # Pop the classes from the clustering(will be inserted later)
          clustering_info[CLASS_INFO_IDX].pop(index_1)
     else:
          # Add the value to the new cluster 
          cluster.append(value_1)
          # Pop the classes from the clustering(will be inserted later)
          clustering_info[CLASS_INFO_IDX].pop(index_1)
     # Retrieve the index information from the clustering for value 2
     index_2 = clustering_info[CLASS_INFO_IDX].index(value_2)
     if type(value_2) is list:
          # Append the values to the new cluster 
          for item in value_2:
              cluster.append(item)  
          # Pop the classes from the clustering(will be inserted later)
          clustering_info[CLASS_INFO_IDX].pop(index_2)
     else:
          # Add the value to the new cluster 
          cluster.append(value_2)
          # Pop the classes from the clustering(will be inserted later)
          clustering_info[CLASS_INFO_IDX].pop(index_2)
     # print("Cluster(before) :", cluster) 
     cluster_size = len(cluster)
     # Update the sealed class counter information if cluster size = k
     if cluster_size == k_size:
          for _class in cluster:
               # Make a copy of the classes from the QID info that belong to the generated cluster
               # Needed for reverting the classes at the end of each combination
               if curr_qid_cluster is not None:
                    curr_qid_cluster[_class] = copy.deepcopy(qid_info[_class])
               # Update the QID information of the classes that belong to the cluster with the original information
               # Original infomation is needed to calculate the neigboring distance
               qid_info[_class] = copy.deepcopy(dataset_info[QID_INFO_IDX][_class])
          sealed_cntr = clustering_info[SEALED_COUNTER_INFO_IDX]
          sealed_cntr += 1
          clustering_info[SEALED_COUNTER_INFO_IDX] = sealed_cntr
          # Insert the new cluster into the clustering at the start  
          clustering_info[CLASS_INFO_IDX].insert(0, cluster)
     # Update both cluster and sealed class counter information if cluster size > k
     elif cluster_size > k_size: 
          for _class in cluster:
               # Make a copy of the classes from the QID info that belong to the generated cluster
               # Needed for reverting the classes at the end of each combination
               if curr_qid_cluster is not None:
                    curr_qid_cluster[_class] = copy.deepcopy(qid_info[_class])
               # Update the QID information of the classes that belong to the cluster with the original information
               # Original infomation is needed to calculate the neigboring distance
               qid_info[_class] = copy.deepcopy(dataset_info[QID_INFO_IDX][_class])
          # Calculate the nearest neighbors with respect to a pivot class 
          # Sorted list of classes is returned 
          cluster = calculate_nearest_neighbors(cluster, qid_info)
          # Partition the cluster into two clusters based on the distance 
          #     Cluster with size = k classes insert at the start 
          #     Cluster with the remaining classes append at the end 
          clustering_info[CLASS_INFO_IDX].insert(0, cluster[0:k_size])
          clustering_info[CLASS_INFO_IDX].append(cluster[k_size:])
          sealed_cntr = clustering_info[SEALED_COUNTER_INFO_IDX]
          sealed_cntr += 1
          clustering_info[SEALED_COUNTER_INFO_IDX] = sealed_cntr
          # Generalize the partitioned values 
          if len(cluster[k_size:]):
               generalize_cluster_info(cluster[k_size:], qid_info)
     # Update cluster information if cluster size < k
     else:
          for _class in cluster:
               # Make a copy of the classes from the QID info that belong to the generated cluster
               # Needed for reverting the classes at the end of each combination
               if curr_qid_cluster is not None:
                    curr_qid_cluster[_class] = copy.deepcopy(qid_info[_class])
          # Append the new cluster into the clustering at the end
          clustering_info[CLASS_INFO_IDX].append(cluster)
     # print("Cluster(after) :", cluster)
     # print("Clustering Info(after) :", clustering_info)
     generate_end_time = time.time() 
     # print("Generate clustering execution time :", generate_end_time - generate_start_time, "secs\n")
     # print("#################################################")
     return cluster[0:k_size], cluster[k_size:]

# Test case - 1  
# k-size = 2
# Input - clstr_info = [[0, 1, 2, 3, 4, 5, 6], 0]
# Output - clstr_info = [[[0, 4], 1, 2, 3, 5, 6], 1]
#          Sealed counter = 1 
#k_size = 2
#clstr_info = [[0, 1, 2, 3, 4, 5, 6], 0]
#generate_clustering_info(0, 4, clstr_info, k_size)

# Test case - 2 
# k-size = 2
# Input - clstr_info = [[[0, 1], 2, 3, 4, 5, 6], 0]
# Output - clstr_info = [[[0, 1], 2, 3, 5, 6, [4]], 1]
#          Sealed counter = 1 
#k_size = 2
#clstr_info = [[[0, 1], 2, 3, 4, 5, 6], 0]
#generate_clustering_info([0, 1], 4, clstr_info, k_size)

# Test case - 3 
# k-size = 3
# Input - clstr_info = [[0, 1, 2, 3, 4, 5, 6], 0]
# Output - clstr_info = [[[0, 1], 2, 3, 4, 5, 6], 0]
#          Sealed counter = 0 
#k_size = 3
#clstr_info = [[0, 1, 2, 3, 4, 5, 6], 0]
#generate_clustering_info(0, 1, clstr_info, k_size)

# Test case - 4 
# k-size = 3
# Input - clstr_info = [[[0, 1], [2, 3], 4, 5, 6], 0]
# Output - clstr_info = [[[0, 1, 2], 4, 5, 6, [3]], 1]
#          Sealed counter = 1 
#k_size = 3
#clstr_info = [[[0, 1], [2, 3], 4, 5, 6], 0]
#generate_clustering_info([0, 1], [2, 3], clstr_info, k_size)



# Revert the clustering changes for next combination within the iteration 
def revert_clustering_info(class_info, clustering_info, class_size, k_size):
     revert_start_time = time.time() 
     # print("########################## Revert clustering information")
     # print("Class info :", class_info)
     # print("Clustering info(before) :", clustering_info)
     # print("Class size :", class_size)
     # print("K-size :", k_size)
     # Classes requested to be clustered are greater than k-size
     if class_size == k_size:
          # Delete the first element in the list that is sealed 
          del clustering_info[CLASS_INFO_IDX][0]
          # Decrement the sealed counter 
          clustering_info[SEALED_COUNTER_INFO_IDX] -= 1 
     # Classes requested to be clustered are greater than k-size
     elif class_size > k_size:
          # Delete the first element in the list that is sealed 
          del clustering_info[CLASS_INFO_IDX][0]
          # Delete the last element in the list that is unsealed
          del clustering_info[CLASS_INFO_IDX][-1]
          # Decrement the sealed counter 
          clustering_info[SEALED_COUNTER_INFO_IDX] -= 1 
     # Classes requested to be clustered are lesser than k-size
     else:
          # Delete the last element in the list that is unsealed
          del clustering_info[CLASS_INFO_IDX][-1]
     # Add the initial classes information to the clustering at those indices 
     for _index in class_info.keys(): 
          clustering_info[CLASS_INFO_IDX].insert(_index, class_info[_index])
     # print("Clustering info(after) :", clustering_info)
     revert_end_time = time.time() 
     # print("Revert clustering execution time :", revert_end_time - revert_start_time, "secs\n")
     # print("#################################################")



# Generate the clustering with minimum preserve 
def generate_min_preserve_clustering(dataset_name, dataset_info, k_size, threshold):
     print("##########################  Generate minimum preserve clustering")
     iteration_cntr = 0
     changed_pattern_set = set()
     best_prev_greater_preserve_patterns = {}
     # Clustering information to generate combinations 
     combination_info = copy.deepcopy(dataset_info[CLUSTERING_INFO_IDX][CLASS_INFO_IDX])
     # Clustering info 
     clustering_info = copy.deepcopy(dataset_info[CLUSTERING_INFO_IDX])
     # QID information 
     qid_info = copy.deepcopy(dataset_info[QID_INFO_IDX])
     # Calculate the classes distances with other classes based on the FDs
     # Combinations are generated with nearest classes based on the configured threshold 
     classes_distance = calculate_classes_distance(dataset_name, qid_info)
     # Pattern information 
     #         Starts the combination with the seed value from START for preserve and best classes  
     #         Calculates new preserve and best classes and if CGU is greater update the BEST for preserve and best classes   
     curr_pattern_info = copy.deepcopy(dataset_info[PATTERN_INFO_IDX])
     while True: 
          max_cgu = 0 
          best_qid_cluster = {}
          best_class_info = [] 
          iter_start_time = time.time()        
          combinations = {}
          best_sealed_cluster = []
          best_curr_greater_preserve_patterns = {}
          COMBINATION_GENERATED = False 
          # print("Remaining classes :", len(classes_distance.keys()))
          # Generate the combinations for the classes in the clustering (excluding sealed classes)
          for class_i in list(classes_distance.keys()): 
               _class_cntr = 0
               _threshold_cntr = 0
               _first_duplicate = 0
               CLASS_THRESHOLD_SET = False  
               prev_class_distance = -1
               curr_class_distance = -1 
               class_distance_i = classes_distance[class_i]
               # Check if the class that is being used to generate the combination is sealed 
               if class_distance_i[SEALED_CLASS_INFO_IDX] != 0:
                    del classes_distance[class_i] 
                    # print("Deleted sealed class_i : %d" %(class_i))
                    continue
               # Iterate through the nearest neighbor classes for the configured threshold values
               while True: 
                    comb_start_time = time.time()        
                    curr_qid_cluster = {} 
                    greater_preserve_patterns = {} 
                    # Check if the threashold has reached for the class 
                    if _threshold_cntr >= threshold or _class_cntr >= len(class_distance_i[CLASSES_DISTANCE_INFO_IDX]):
                         # print("###################### Threshold reached for class :", class_i)
                         # Indicate that the class threshold has been reached (so we skip some operations)
                         CLASS_THRESHOLD_SET = True  
                         break
                    # Get the closest class information w.r.t class being compared
                    try:
                         class_j, curr_class_distance = class_distance_i[CLASSES_DISTANCE_INFO_IDX][_class_cntr]
                         class_distance_j = classes_distance[class_j]
                    except:
                         del class_distance_i[CLASSES_DISTANCE_INFO_IDX][_class_cntr]
                         _class_cntr += 1
                         # print("Skipping combination (%d, %d) - Deleted classes" %(class_i, class_j))
                         continue
                    # Check if the class that is being used to generate the combination 
                    #    * Sealed [Valid]
                    #    * Same distance as the previous class [Invalid]
                    #    * Being repeated [Valid]
                    #    * Duplicate tuples (other than the first one) [Valid] 
                    # Skip that class and process the next class.
                    # print("Class(%d) distances :" %(class_i), class_distance_i)
                    # print("Class(%d) distances :" %(class_j), class_distance_j)
                    # print("Threshold cntr(%d), Class cntr(%d)" %(_threshold_cntr, _class_cntr))
                    try: 
                         indx_i = class_distance_i[CLUSTERED_CLASS_INFO_IDX]
                         indx_j = class_distance_j[CLUSTERED_CLASS_INFO_IDX]
                         # Sealed classes 
                         if (class_distance_j[SEALED_CLASS_INFO_IDX] != 0):
                              del class_distance_i[CLASSES_DISTANCE_INFO_IDX][_class_cntr]
                              _class_cntr += 1
                              # print("Skipping combination (%d, %d) - Sealed classes" %(class_i, class_j))
                              continue
                         # Equal distant clases  
                         # else if (curr_class_distance == prev_class_distance):  
                         #     del classes_distance[_class_cntr] 
                         #     _class_cntr += 1
                         #     # print("Skipping combination (%d, %d) - Equal disant classes" %(class_i, class_j))
                         #     continue
                         # After the combination validation update the indices to the clustered class information 
                         # Algorithm below needs the clustered information at the indicies generated during subsequent iterations
                         # Skip the combination if the classes belong to the same cluster
                         # indx_i = cluster(2, 5), indx_j = cluster(2, 5) 
                         # Avoids generating a combination of ((2, 5), (2, 5))
                         if indx_i == indx_j:
                              del class_distance_i[CLASSES_DISTANCE_INFO_IDX][_class_cntr]
                              _class_cntr += 1
                              # print("Skipping combination (%d, %d) - Same clusters" %(class_i, class_j))
                              continue
                         # Repeated classes(combinations)
                         if combinations[str(indx_i) + "_" + str(indx_j)]:
                              del class_distance_i[CLASSES_DISTANCE_INFO_IDX][_class_cntr]
                              _class_cntr += 1
                              # print("Skipping combination (%d, %d) - Repeated classes" %(class_i, class_j))
                              continue
                         elif combinations[str(class_i) + "_" + str(class_j)]:
                              del class_distance_i[CLASSES_DISTANCE_INFO_IDX][_class_cntr]
                              _class_cntr += 1
                              # print("Skipping combination (%d, %d) - Repeated classes" %(class_i, class_j))
                              continue
                         # Duplicate classes (skip other than the first duplicate)
                         if ((curr_class_distance == prev_class_distance) and (qid_info[class_i] == qid_info[class_j])):
                              # Check if it is the second duplicate
                              if _first_duplicate != 0:
                                   del class_distance_i[CLASSES_DISTANCE_INFO_IDX][_class_cntr]
                                   _class_cntr += 1
                                   # print("Skipping combination (%d, %d) - Duplicate classes" %(class_i, class_j))
                                   continue
                              # Mark that we have seen the first duplicate 
                              _first_duplicate = 1
                              # print("Combination (%d, %d) - First duplicate classes (Don't skip)" %(class_i, class_j))
                    except: 
                         # Update the previous distance with the current distance
                         prev_class_distance = curr_class_distance
                         # Increment the threshold and class counter as the combination is successfully generated
                         _threshold_cntr += 1
                         _class_cntr += 1 
                         # Update the generated combination in the dictionary so we don't generate it again
                         # Example : [0,1] is seen then we don't generate [1,0] and vice versa 
                         combinations[str(class_i) + "_" + str(class_j)] = 1
                         combinations[str(class_j) + "_" + str(class_i)] = 1
                         # Update the generated combination in the dictionary so we don't generate it again
                         # Example : ([0,1], [2,3]) are seen then we don't generate ([2,3], [0,1]) and vice versa
                         combinations[str(indx_i) + "_" + str(indx_j)] = 1
                         combinations[str(indx_j) + "_" + str(indx_i)] = 1
                    print("------> Iteration : %d Combination : (" %(iteration_cntr), class_i, ",", class_j, ")\n")
                    # Indicate the the combination has been generated (set the flag)
                    COMBINATION_GENERATED = True 
                    combination_key = str(indx_i) + "_" + str(indx_j)
                    # Performance optimization - computing the preserve values from already stored values for the 
                    # combination in the last iteration and updating the same with the best combinations preseve
                    # values from the last iteration
                    try:
                         cached_preserve = 0.0
                         # Retrieve the cached combination information (during iteration-0 the cache would be built)
                         # print("Combination cache key :", combination_key)
                         cached_combination_data = combinations_cache[combination_key]
                         # print("Combination patterns + preserve info :", cached_combination_data)
                         # Stored CGU information 
                         cached_cgu = cached_combination_data[COMBINATION_CURR_CGU_INFO] 
                         cached_preserve = cached_cgu * len(curr_pattern_info)
                         # print("Combination cached preserve(CGU) : %f(%f)" %(cached_preserve, cached_cgu))
                         # print("Best combination patterns + preserve info :",  best_prev_greater_preserve_patterns)
                         # Retrieve the patterns with greater preserve from last iteration 
                         comb_patterns = set(cached_combination_data[GREATER_PRESERVE_PATTERNS].keys())
                         best_patterns = set(best_prev_greater_preserve_patterns.keys())
                         # Get the overlapped patterns between the combination and best preseve patterns from last iteration
                         overlap_patterns = comb_patterns.intersection(best_patterns) 
                         # print("Overlap patterns :", overlap_patterns)
                         # Get the disjoint patterns from the best preseve patterns obtained in the last iteration
                         disjoint_patterns = best_patterns.difference(overlap_patterns)
                         # print("Disjoint patterns :", disjoint_patterns)
                         # Retrieve the patterns + preserve data with greater preserve from last iteration 
                         comb_patterns = cached_combination_data[GREATER_PRESERVE_PATTERNS] 
                         # Update the combination CGP information with the best patterns information found in previous iteration
                         # Disjoint patterns just update the values with the new values after removing the older values 
                         for _cached_pattern in disjoint_patterns:
                              # Best pattern current preserve value obtained in the last iteration 
                              best_pattern_curr_preserve = best_prev_greater_preserve_patterns[_cached_pattern][0]
                              # Best pattern previous preserve value obtained in the last iteration 
                              best_pattern_prev_preserve = best_prev_greater_preserve_patterns[_cached_pattern][1]
                              # Updated the cached preserve value 
                              cached_preserve -= best_pattern_prev_preserve    
                              cached_preserve += best_pattern_curr_preserve    
                              # Update the internal pattern preserve also for the combination
                              curr_pattern_info[_cached_pattern][PATTERN_INFO_METRIC_CURRENT_IDX] = best_pattern_curr_preserve
                         # Overlapping patterns need to update with the max (combination value, best patterns)
                         # Update the cached patterns with the updated value if best preserve patterns from previous iteration is greater
                         for _cached_pattern in overlap_patterns:
                              # Best pattern current preserve value obtained in the last iteration 
                              best_pattern_curr_preserve = best_prev_greater_preserve_patterns[_cached_pattern][0]
                              # Combination current preserve value obtained in the last iteration 
                              comb_pattern_curr_preserve = comb_patterns[_cached_pattern][GREATER_PRESERVE_PATTERNS_CURR_PRESERVE]  
                              # Update only if the best pattern preserve is greater than the combination pattern preserve 
                              if best_pattern_curr_preserve > comb_pattern_curr_preserve:
                                   cached_preserve -= comb_pattern_curr_preserve    
                                   cached_preserve += best_pattern_curr_preserve    
                                   # Update the internal pattern preserve also for the combination
                                   comb_patterns[_cached_pattern][GREATER_PRESERVE_PATTERNS_CURR_PRESERVE] = best_pattern_curr_preserve
                                   comb_patterns[_cached_pattern][GREATER_PRESERVE_PATTERNS_PREV_PRESERVE] = comb_pattern_curr_preserve
                         # Store the calculated CGU for the combination 
                         cached_cgu = float(cached_preserve)/len(curr_pattern_info)                         
                         # print("Updated combination cached preserve(CGU) : %f(%f)" %(cached_preserve, cached_cgu))
                         cached_combination_data[COMBINATION_CURR_CGU_INFO] = cached_cgu                       
                         # Validate the cached CGU information with the maximum CGU value
                         if max_cgu != 0 and cached_cgu < max_cgu:
                              print("\n Current Cached CGU(%f) <= MAX CGU(%f)\n" %(cached_cgu, max_cgu))
                              comb_end_time = time.time() 
                              print("\n\nCombination execution time :", comb_end_time - comb_start_time, "secs\n")
                              continue
                    except Exception as e:
                         # print("Exception :", e) 
                         pass
                    # Pattern information (gets updated at the end of the iteration with best one)
                    # Revert only the changed patterns from previous combination to the start values 
                    for _key in changed_pattern_set:
                         pattern_data = curr_pattern_info[_key]
                         pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX] = pattern_data[PATTERN_INFO_METRIC_START_IDX]
                         pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX] = pattern_data[PATTERN_INFO_BEST_CLASSES_START_IDX]  
                    # print("Start Pattern Info:")
                    # print(curr_pattern_info)
                    # Clear the changed pattern set after updating the information 
                    changed_pattern_set.clear()
                    # Store the class information prior to generating clustering (revert back after the combination)
                    curr_class_info, curr_class_size = store_clustering_info(indx_i, indx_j, clustering_info)
                    # print("\nOriginal clustering info :\n", clustering_info)
                    # print("\nOriginal QID info :\n", qid_info, "\n")
                    # Generate clustering based on the combination information 
                    cluster, partition = generate_clustering_info(indx_i, indx_j, clustering_info, qid_info, curr_qid_cluster, k_size)  
                    # print("\nCalculated clustering info :\n", clustering_info)
                    # print("\nActual QID cluster info(ungeneralized) :\n", curr_qid_cluster)
                    # Generalize the QID information within the generated cluster prior to calculating the utility   
                    generalize_cluster_info(cluster, qid_info)
                    # print("\nGeneralized QID info :\n", qid_info)
                    # Calculate the preserve for all the patterns with respect to a clustering
                    calculate_patterns_preserve_clustering(clustering_info, curr_pattern_info, changed_pattern_set, greater_preserve_patterns, qid_info, k_size) 
                    # print("Current Pattern Info:")
                    # print(curr_pattern_info)
                    # Calculate the Clustering Generalization Utility(CGU) metric based on overall pattern information 
                    curr_cgu = calculate_clustering_generalization_utility(curr_pattern_info) 
                    # Compare the current CGU with the stored MAX CGU and update accordingly
                    if (curr_cgu > max_cgu):
                         best_qid_cluster = {}
                         best_class_info = [] 
                         print("\nCurrent CGU(%f) > MAX CGU(%f) - Updated\n" %(curr_cgu, max_cgu))
                         max_cgu = curr_cgu
                         # Store the best pattern info with highest CGU for a particular combination within the iteration
                         # Update the best pattern info with only the changed preserve and best classes within the combination
                         best_pattern_set = set(changed_pattern_set)
                         for _key in best_pattern_set:
                              pattern_data = curr_pattern_info[_key]
                              pattern_data[PATTERN_INFO_METRIC_BEST_IDX] = pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX]
                              pattern_data[PATTERN_INFO_BEST_CLASSES_BEST_IDX] = pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX]  
                         # print("Best Pattern Info:")
                         # print(curr_pattern_info)
                         # Store the best class index information which will be used to restore later 
                         best_class_info.append(indx_i)  
                         best_class_info.append(indx_j)  
                         # print("\nBest class info indx : (", indx_i, ",", indx_j, ")")
                         # Keep track of best QID info cluster within the iteration(generalized)
                         # Best QID cluster information stored would be restored at the end of the iteration 
                         for _class in cluster + partition:
                              best_qid_cluster[_class] = copy.deepcopy(qid_info[_class])
                         # print("\nBest QID cluster info(generalized) :\n", best_qid_cluster, "\n")
                         # Store the best sealed cluster information for the iteration only if the cluster size is K. Stored information
                         # is used at end of the iteration to retrieve patterns from the sealed cluster information whose preserve value 
                         # has reached 1. These patterns are cached and used in subsequent combinations to skip pattern computations. 
                         best_sealed_cluster = []
                         if len(cluster) == k_size:
                              best_sealed_cluster = copy.deepcopy(cluster)    
                         # Update the best greater patterns preserve 
                         best_curr_greater_preserve_patterns = copy.deepcopy(greater_preserve_patterns)  
                    else:
                         print("\nCurrent CGU(%f) <= MAX CGU(%f)\n" %(curr_cgu, max_cgu))
                         pass 
                    # Revert the clustering changes for next combination within the iteration 
                    revert_clustering_info(curr_class_info, clustering_info, curr_class_size, k_size)
                    # print("\nRevert clustering info :\n", clustering_info)
                    # Revert generalization changes made to QID info for next combination within the iteration
                    for _item in curr_qid_cluster.keys():
                         qid_info[_item] = curr_qid_cluster[_item]
                    # print("\nRevert QID info :\n", qid_info)
                    # Cache the combination information 
                    combinations_cache[combination_key] = [greater_preserve_patterns, curr_cgu]
                    # print("Update combination patterns + preserve info :", combinations_cache[combination_key])
                    comb_end_time = time.time() 
                    print("\n\nCombination execution time :", comb_end_time - comb_start_time, "secs\n")
               # Continue to generate combinations with the remaining classes (within the iteration)   
               if CLASS_THRESHOLD_SET:
                    continue
          # No combinations were generated in the iteration (terminate the algorithm) 
          if COMBINATION_GENERATED is False:
               print("###################### No combinations were generated in the iteration (terminate the algorithm)")
               break
          # Update the the previous best pattern preserve with the one found in this iteration 
          best_prev_greater_preserve_patterns = best_curr_greater_preserve_patterns
          # Restore the clustering information with the best clustering index information stored in the iteration 
          cluster, partition = generate_clustering_info(best_class_info[0], best_class_info[1], clustering_info, qid_info, None, k_size)  
          # print("\nRestore clustering info :\n", clustering_info)
          # Update the QID information with the best QID cluster information(generalized) found in the iteration 
          for _item in best_qid_cluster.keys():
               qid_info[_item] = best_qid_cluster[_item]
          # print("\nRestore QID cluster(generalized) :\n", qid_info)
          # Update the clustering information
          print("\n\nBest CGU info (iteration %d) :" %(iteration_cntr), max_cgu)
          # Compare the calculated CGU with the cut-off threshold (optimization)
          if (max_cgu >= MAX_DEPENDENCY_GEN_CGU_THRESHOLD):
               print("Calculated CGU(%f) > cut-off threshold(%f)\n" %(max_cgu, MAX_DEPENDENCY_GEN_CGU_THRESHOLD))
               break
          # print("Best clustering info (iteration %d) :\n" %(iteration_cntr), clustering_info)
          # print("Best qid info (iteration %d) :\n" %(iteration_cntr), qid_info)
          # print("Best pattern info (iteration %d) :\n" %(iteration_cntr), best_pattern_info)
          # Revert only the changed patterns from previous combination to the start values 
          for _key in changed_pattern_set:
               pattern_data = curr_pattern_info[_key]
               pattern_data[PATTERN_INFO_METRIC_CURRENT_IDX] = pattern_data[PATTERN_INFO_METRIC_START_IDX]
               pattern_data[PATTERN_INFO_BEST_CLASSES_CURRENT_IDX] = pattern_data[PATTERN_INFO_BEST_CLASSES_START_IDX]  
          # Update the pattern information in the dataset with the best pattern information found within the iteration 
          # Only the patterns whose preserve and best classes changed would be updated (acts as seed for next iteration)
          for _key in best_pattern_set:
               pattern_data = curr_pattern_info[_key]
               pattern_data[PATTERN_INFO_METRIC_START_IDX] = pattern_data[PATTERN_INFO_METRIC_BEST_IDX]
               pattern_data[PATTERN_INFO_BEST_CLASSES_START_IDX] = pattern_data[PATTERN_INFO_BEST_CLASSES_BEST_IDX]  
          # print("End Pattern Info:")
          # print(curr_pattern_info)
          # Update the changed pattern set, so it updates the current values with the start for the next iteration
          changed_pattern_set = best_pattern_set 
          # print("Best sealed cluster (iteration):", best_sealed_cluster)
          # Generate the patterns within the best sealed cluster
          original_qid_info = dataset_info[QID_INFO_IDX]
          for _class in best_sealed_cluster:
               qid_data = original_qid_info[_class]
               # Iterate through the FDs to generate keys to store pattern information
               for item in FD_INFO:
                    pattern_key = None
                    for i in item:
                         if pattern_key is None:
                              pattern_key = qid_data[i]
                         else:
                              pattern_key = pattern_key + " " + qid_data[i]
                    # Add the generated pattern to the cache set only for patterns whose preserve has reached maximum(1)
                    pattern_data = curr_pattern_info[pattern_key]
                    # print("Pattern : %s, Preserve : %f" %(pattern_key, pattern_data[PATTERN_INFO_METRIC_START_IDX]))
                    if pattern_data[PATTERN_INFO_METRIC_START_IDX] == 1.0:
                         sealed_patterns_cache.add(pattern_key)
          # print("Sealed cluster patterns :", sealed_patterns_cache)
          # Clustering size information (changes when classes get clustered)
          clustering_size = len(clustering_info[CLASS_INFO_IDX]) 
          # print("\n\nClustering size : %d" %(clustering_size))
          # Sealed classes information  
          sealed_cntr = clustering_info[SEALED_COUNTER_INFO_IDX]
          # print("Sealed class counter : %d\n" %(sealed_cntr))
          # Stop the process for classes less than K
          if (clustering_size - sealed_cntr) < k_size:
               class_size = 0 
               # Unsealed clusters and classes can be clustered as well
               # Following example, unsealed classes and clusters [3, 6] and [8] can be clustered
               # K= 3, [[[0, 1, 5], [2, 4, 7], [3, 6] [8]], 2]
               for _item in range(clustering_size - sealed_cntr):
                    if type(clustering_info[CLASS_INFO_IDX][sealed_cntr + _item]) is list:
                         class_size += len(clustering_info[CLASS_INFO_IDX][sealed_cntr + _item])      
                    else:
                         class_size += 1
               # print("\nUnsealed class size :", class_size, "\n")
               if class_size < k_size:
                    break 
          # Generate the combinations for the next iteration
          # Update the classes distance information with the sealed clustered classes(so we don't generate combinations)
          for _class in cluster:
               class_distance = classes_distance[_class]     
               # Update the cluster information 
               class_distance[CLUSTERED_CLASS_INFO_IDX] = cluster
               if len(cluster) == k_size:
                    class_distance[SEALED_CLASS_INFO_IDX] = 1
          # Update the classes distance information with the partitioned clustered classes(so we do generate combinations)
          for _class in partition:
               class_distance = classes_distance[_class]     
               # Update the partition information 
               class_distance[CLUSTERED_CLASS_INFO_IDX] = partition
               class_distance[SEALED_CLASS_INFO_IDX] = 0
          # print("\nUpdated classes distance info :\n", classes_distance)
          iter_end_time = time.time() 
          print("\n\nIteration [%d] execution time :" %(iteration_cntr), iter_end_time - iter_start_time, "secs\n\n\n")
          iteration_cntr += 1
     return clustering_info, curr_pattern_info, qid_info
     # print("#################################################")



if __name__ == '__main__':
     # Parse the command line arguments for dataset and dataset hierarchy names 
     if 4 == len(sys.argv):
          # Dataset name 
          dataset_name = sys.argv[1]
          # print("Dataset name : %s" %(dataset_name))
          # K-anonymity value  
          k_size = int(sys.argv[2])
          # print("K-anonymity size : %d" %(k_size))
          if k_size < 2:
               # print("K-anonymity size should be atleast 2")
               exit(0)
          # Threshold value  
          threshold = int(sys.argv[3])
          # Obtain the parsed dataset information 
          dataset_info = get_parsed_dataset_info(dataset_name, k_size)
          try:
               # Retrieve the cached hierrachy information of nodes 
               hierarchy_file = (dataset_name.split('.csv'))[0] + '_hierarchy.pickle'
               hierarchy_fd = open(hierarchy_file, 'rb')
               hierarchy_cache = pickle.load(hierarchy_fd)
               print("Hierarchy information found in the pickled file -", hierarchy_file) 
               hierarchy_fd.close()
          except:
               # Do nothing 
               pass 
          try:
               # Retrieve the cached preserve values of the patters with respect to a clustered class
               preserve_file = (dataset_name.split('.csv'))[0] + '_preserve.pickle'
               preserve_fd = open(preserve_file, 'rb')
               patterns_preserve_cluster_cache = pickle.load(preserve_fd)
               print("Pattern preserve information found in the pickled file -", preserve_file) 
               preserve_fd.close()
          except:
               # Do nothing
               pass 
          # Generate the clustering with maximum utility 
          match = re.search(r'data/', dataset_name)
          if match is None:
               dataset_name = "data/" + dataset_name
               dataset_qid_name = dataset_name + ".qid"
          start_time = time.time()        
          clustering_info, pattern_info, qid_info = generate_min_preserve_clustering(dataset_name, dataset_info, k_size, threshold)
          # print("Optimal clustering information : \n", clustering_info[CLASS_INFO_IDX], "\n\n")
          # Store clustered and generalized QID information  
          store_clustered_qid_info(dataset_name, clustering_info, pattern_info, qid_info, k_size) 
          end_time = time.time() 
          # Cache the calculated hierarchy information  
          hierarchy_fd = open(hierarchy_file, 'wb')
          pickle.dump(hierarchy_cache, hierarchy_fd)
          hierarchy_fd.close()
          # Cache the calculated preserve pattern information  
          preserve_fd = open(preserve_file, 'wb')
          pickle.dump(patterns_preserve_cluster_cache, preserve_fd)
          preserve_fd.close()
          print("\nOver all execution time of", __file__, ":", end_time - start_time, "secs\n")
     else:
          # Incorrect command line arguments passed 
          # print("Usage:")
          print("python <program_name.py> <dataset_name> <k-anonymity_size> <threshold_value>")
          pass
