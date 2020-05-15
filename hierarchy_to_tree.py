import json 
import ast

global noParent #variable holding the string used when a node does not have a parent
noParent = "None"

def get_children(node, hierarchy): #http  s://stackoverflow.com/questions/18025130/recursively-build-hierarchical-json-tree
    return [x[1] for x in hierarchy if x[0] == node] #finds the child of the node



def get_nodes(node, hierarchy): #https://stackoverflow.com/questions/18025130/recursively-build-hierarchical-json-tree
    d = {}
    d['value'] = node #gets the name of the node
    parent = [x[0] for x in hierarchy if x[1] == node] #gets the parent value of the node from the links list
    if parent == []: d['parent'] = noParent #if the node has no parent, "None" is used instead
    else: d['parent'] = parent[0] #otherwise, write the parent of the node
    children = get_children(node, hierarchy)
    if children:
        d['children'] = [get_nodes(child, hierarchy) for child in children] #finds the children of the current node
    else:
        d['children'] = [] #if the node has no children, it is shown as an empty list
    return d



def json_tree(hierarchy): #https://stackoverflow.com/questions/18025130/recursively-build-hierarchical-json-tree
    # print("Hierarchy :\n", hierarchy)
    parents, children = zip(*hierarchy) #gets all parent and child values
    # print("Parents :\n", parents)
    # print("Children :\n", children)
    root_node = {x for x in parents if x not in children} #the root node is the one without a parent
    # print("Root node :\n", root_node)
    hierarchy.append(('value', root_node))
    tree = get_nodes(list(root_node)[0], hierarchy) #creates the tree
    # print("Tree :\n", tree)
    return json.dumps(tree, indent=4) #writes the tree to the json file



def generate_trees(linksList):
    jsonList = []
    for file in linksList:
        with open(file, "r") as f: #reads the file containing the parent-child links
            hierarchy = ast.literal_eval(f.read())
        gettingName = file.split("/") #gets the name of the hierarchy
        fileName = gettingName[-1][:-4]
        hierarchyFile = fileName + ".json" #turns the hierarchy into a JSON file
        jsonList.append(hierarchyFile)
        jsonFile = "data/" + hierarchyFile
        with open(jsonFile, "w") as jfile: #opens the JSON file and writes the hierarchy in the format of a JSON tree
            tree = json_tree(hierarchy)
            # print("JSON tree :\n", tree)
            jfile.write(tree)
    return jsonList



if __name__ == '__main__':
    HIERARCHY_LINKS = ["./data/patient/state.txt"]
    # Generate the JSON trees from the hierarchy links
    json_trees_list = generate_trees(HIERARCHY_LINKS) 
    # print(json_trees_list)
