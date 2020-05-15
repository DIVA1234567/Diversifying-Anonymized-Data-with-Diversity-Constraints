

def parse_single_constraint(s):
    '''
    Args: String constraint
    format as: s = "gen(male) [100, 1000]"
    Return:
        dict, with attr, val, lower_bound, up_bound as key
    '''
    import re
    m = re.match(r'(.*)\((.*)\) \[(\d+), (\d+)\]', s)
    if m:
        d = dict()
        d['attr'], d['val'], d['lower_bound'], d['up_bound'] = m.group(1), m.group(2),m.group(3),m.group(4)
        return d
    else:
        return None

# test
# s = "gen(male) [100, 1000]"
# d = parse_single_constraint(s)
# print(d)

def parse_constraint_file(path):
    '''
    read constraints from file 
    Return:
        a list of dict, each dict is a constraint
    '''
    constraints = []
    with open(path) as f:
        lines = f.readlines()
        for l in lines:
            constraints.append(parse_single_constraint(l))
    return constraints

# test
# path = 'C:/Users/datasci/YuHuang/YuMcMasterDropbox/Dropbox/Huang_PaperReview/Huang_3rd_research/Huang_Data_Fairness/Huang_Data_Fairness_Experiment/Huang_code/constraint.txt'
# l = parse_constraint_file(path)
# print(l)


class Node:
    '''
    One Node reprensent a constraint and the tuples
    that satify this constraints
    self.records is a list of tuples satisfy this constraint
    '''
    def __init__(self, id, constraint):
        self.id = id
        self.constraint = constraint
        # records satisfy the constraint
        # store records ids
        self.records = []
        self.neighbors = []

# test Node   
# n = Node(1, d)
# n.constraint
# node_li = []
# for i in range(len(l)):
#     node_li.append(Node(i, l[i]))
# node_li[2].constraint


def build_constraint_node_list(path):
    node_li = []
    constraints_li = parse_constraint_file(path)
    for i in range(len(constraints_li)):
        node_li.append(Node(i,constraints_li[i]))

    return node_li

# test
# node_l = build_constraint_node_list(path)
# node_l[1].constraint


def build_graph(node_li):
    '''
    if two nodes have overlap on the satisfied records
    then add them to each other neighbor list
    '''
    for i in range(len(node_li)):
        for j in range(i+1, len(node_li)):
            node_i, node_j = node_li[i], node_li[j]
            overlap_records = set(node_i.records) & set(node_j.records)
            if overlap_records:
                node_i.neighbors.append(node_j)
                node_j.neighbors.append(node_i)

    return node_li

# test
# node_1 = Node(1,{})
# node_1.records = [1,2,3]
# node_2 = Node(2,{})
# node_2.records = [2,4]
# node_12 = [node_1, node_2]
# node_new_12 = build_graph(node_12)
# node_new_12[0].neighbors[0].id