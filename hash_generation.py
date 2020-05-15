import json
import os
import math 
import pandas as pd

def depth(data): #returns depth of the tree
    #function found here: https://stackoverflow.com/questions/29005959/depth-of-a-json-tree
    if 'children' in data:
        return 1 + max([-1] + list(map(depth, data['children'])))
    else: return 1



def node_depths(data, currDepth, nDepths): #creates dictionary containing information on the depth of each node in the tree
    for val in data:
        currVal = data[val] #get the current value
        if val == "value":
            nDepths[currVal] = currDepth #appends the node and its depth to dictionary nDepths
        elif val == "children" and currVal != []:
            for child in currVal:
                node_depths(child, currDepth - 1, nDepths) #continues to go through the tree
        else: pass;



def node_name(node): #nodes names cannot have underscores in them
    underscores = [i for i, letter in enumerate(node) if letter == "_"] #finds index of every underscore
    if underscores == []: return node
    else: return node[(underscores[-1] + 1):] #everything after the last underscore is the value of the node



def get_bottom_children(data, node, bottomNodes, nDepths): #finds all children at level 0
    maxLevel = nDepths[node_name(node)]
    for val in data:
        currVal, kids = data["value"], data["children"]
        currLevel = nDepths[node_name(currVal)] #find what level of the tree we are currently at
        #if we are too high in the tree, or we've found the node
        if currLevel > maxLevel or (currLevel == maxLevel and currVal == node_name(node)):
            for kid in kids:
                get_bottom_children(kid, node, bottomNodes, nDepths) #go through the children and get to the bottom of the tree
        elif currLevel < maxLevel:
            if currLevel == 0 and currVal not in bottomNodes: # if we've hit the bottom of the tree
                bottomNodes.append(currVal) #append the node to the list of bottom children
            else:
                for kid in kids:
                    get_bottom_children(kid, node, bottomNodes, nDepths) #go through the children and get to the bottom of the tree
        else: pass;



def info_loss(data, node, nDepths): #find the distance between two nodes, as defined in the paper (subject to change)
    genBottomNodes = [] 
    get_bottom_children(data, node, genBottomNodes, nDepths) #find all children at level 0 of the generalized value
    #the distance is the number of level 0 children of the general value divided by the number of level 0 
    #children of the top level value
    return len(genBottomNodes)



def compute_uncertainity(data, node, nDepths, freq_info): #compute the overall uncertainity(E(v)) value for the parent
    total_freq = 0.0
    uncertainity_ev = 0.0
    genBottomNodes = [] 
    genBottomNodes_freq = {} 
    get_bottom_children(data, node, genBottomNodes, nDepths) #find all children at level 0 of the generalized value
    # print("Node's(%s) leaf children :" %(node), genBottomNodes) 
    # Retrieve the frequency information of all the leaf nodes associated with the parent 
    for value in genBottomNodes:
        try:
            # Get the node frequency information
            value_freq = freq_info[value]
        except:
            # Leaf node present in the hierarchy is not present in the attribute frequency information
            continue
        # Store the frequency for the bottom leaf nodes 
        genBottomNodes_freq[value] = value_freq 
        # Add the frequency information to the total frquency 
        total_freq += value_freq
    # print("Leaf nodes frequency :", genBottomNodes_freq)
    # print("Total leaf nodes frequency :", total_freq)
    # Calculate the overall uncertainity value for a given node 
    for value in genBottomNodes_freq.keys():
        value_freq = genBottomNodes_freq[value]
        uncertainity_ev += -((value_freq/total_freq) * math.log10((value_freq/total_freq))) 
    print("Node(%s) uncertainity(E(v)) :" %(node), uncertainity_ev)
    return uncertainity_ev



def generate_hash(currData, fileName, data, hashTable, nDepths, freq_info): #generates hash containing child-parent information
    # print("##########################  Generate hash") 
    for val in currData:
        currVal = currData[val] 
        parent = currData['parent']
        if val == "value":
            if parent == "None": parent = currVal
            nodeName = fileName + str(currVal)
            distance = info_loss(data, parent, nDepths)
            # print("Parent : %s, Distance : %d" %(parent, distance))
            # Uncertainity(E(v)) is set to zero 
	    #	Leaf nodes of the hierarcy - Distance is zero 
            #   Nodes in the hierarchy that have only one child - Distance is one  
            if distance is 0 or distance is 1:
                 hashTable[nodeName] = ((fileName + parent), distance, 0) #appends the node's parent to the hash 
            else: 
                 # Compute the uncertainity value for a given node  
                 uncertainity_ev = compute_uncertainity(data, parent, nDepths, freq_info)
                 hashTable[nodeName] = ((fileName + parent), distance, uncertainity_ev) #appends the node's parent to the hash 
            doubleNode = fileName + str(currVal) + "_" + str(currVal)
            distance = info_loss(data, currVal, nDepths)
            print("Child : %s, Distance : %d" %(currVal, distance))
            # Uncertainity(E(v)) is set to zero 
	    #	Leaf nodes of the hierarcy - Distance is zero 
            #   Nodes in the hierarchy that have only one child - Distance is one  
            if distance is 0 or distance is 1:
                 hashTable[doubleNode] = (nodeName, distance, 0)
            else: 
                 # Compute the uncertainity value for a given node  
                 uncertainity_ev = compute_uncertainity(data, currVal, nDepths, freq_info)
                 hashTable[doubleNode] = (nodeName, distance, uncertainity_ev)
        elif val == "parent": pass;
        elif val == "children":
            for child in currVal:
                generate_hash(child, fileName, data, hashTable, nDepths, freq_info)
        else: pass;
    # print("#################################################")



def create_hash(json_files, dataset, attr_info):
    hashTable = dict()
    # Generate QI data frame information
    qid_df = pd.read_table(dataset, sep=',', header=None, names=attr_info) 
    # Convert all the attributes in the QID data frame to string types
    # Numeric attributes have a DGH and are treated like strings
    qid_df = qid_df.astype(str)
    for _file in json_files:
        # print(_file)
        fileName = "./data/" + _file
        with open(fileName) as f:
            data = json.load(f)
            # print("JSON %s file data\n" %(fileName), data)
        nDepths = dict()
        node_depths(data, depth(data), nDepths)
        # print("Tree depth\n", depth(data))
        # print("Node depth\n", nDepths)
        # Get the frequency information for the attribute 
        attr = _file.split('.')[0]
        # print("Attribute :", attr)
        freq_info = qid_df[attr]
        freq_info = freq_info.value_counts() 
        # print("Frequency info\n", freq_info)
        fileName = str(_file)[:-5] + "_"
        generate_hash(data, fileName, data, hashTable, nDepths, freq_info)
    # Delete all the generated JSON files 
    for _file in json_files:
        fileName = "./data/" + _file
        os.remove(fileName) 
    return hashTable



if __name__ == '__main__':
    JSON_TREE = ["state.json"]
    DATA_SET = "./data/patient/patient.csv"
    ATT_NAMES = ['age', 'postal-codes', 'state', 'diagnosis', 'medication']
    # Generate the JSON trees from the hierarchy links
    hashTable = create_hash(JSON_TREE, DATA_SET, ATT_NAMES)
    # print(hashTable)

