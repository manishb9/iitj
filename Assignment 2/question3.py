from collections import deque

def rightmostNodes(tree):
    if not tree or tree[0] == 0:
        return []

    result = []
    queue = deque([(1, tree[0])])  # Startting with index = 1 for root node

    while queue:
        levelSize = len(queue)
        rightmostNode = None

        for k in range(levelSize):
            index, value = queue.popleft()
            # The rightmostNode node is the last node at this level
            rightmostNode = value  

            # If exist adding the left child
            leftChildIndex = 2 * index
            if leftChildIndex - 1 < len(tree) and tree[leftChildIndex - 1] != 0:
                queue.append((leftChildIndex, tree[leftChildIndex - 1]))

            # If exist adding the right child
            rightChildIndex = 2 * index + 1
            if rightChildIndex - 1 < len(tree) and tree[rightChildIndex - 1] != 0:
                queue.append((rightChildIndex, tree[rightChildIndex - 1]))

        if rightmostNode is not None:
            result.append(rightmostNode)

    return result

# Storing the user input array as ineteger list
tree = list(map(int, input().strip()[1:-1].split(',')))  

rightmostElements = rightmostNodes(tree)
print(" ".join(map(str, rightmostElements)))  
