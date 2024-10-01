from collections import deque, defaultdict
import ast

def checkTreeValidity(n, treeEdges):
    # Tree Building
    tree = defaultdict(list)
    for edge in treeEdges:
        x, y = edge
        tree[x].append(y)
        tree[y].append(x)

    # Using BFS asssuming root is first node in first tree edge
    root = treeEdges[0][0]  
    level = {root: 0}
    queue = deque([root])

    while queue:
        node = queue.popleft()
        current_level = level[node]
        
        for neighbor in tree[node]:
            if neighbor not in level:
                level[neighbor] = current_level + 1
                queue.append(neighbor)

    for node, lvl in level.items():
        if lvl % 2 == 0:  
            if node % 2 == 0:
                return False
        else:  
            if node % 2 != 0:
                return False

    return True

# Input
n = int(input())
treeEdges = ast.literal_eval(input())

# Determine if the tree satisfies the conditions
result = checkTreeValidity(n, treeEdges)
print(result)
