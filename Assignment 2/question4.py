def isPrimeNumber(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def dfs(node, parent, adj, primes, path, paths):
    path.append(node)
    if len(path) > 1:
        primeNumberCount = sum(1 for n in path if primes[n])
        if primeNumberCount == 1:
            paths.append(list(path))
    for neighbor in adj[node]:
        if neighbor != parent:
            dfs(neighbor, node, adj, primes, path, paths)
    path.pop()

def countSpecialNodePaths(n, edges):
    primes = [False] * (n + 1)
    for i in range(1, n + 1):
        primes[i] = isPrimeNumber(i)
    
    adj = [[] for t in range(n + 1)]
    for x, y in edges:
        adj[x].append(y)
        adj[y].append(x)
    
    paths = []
    for i in range(1, n + 1):
        dfs(i, -1, adj, primes, [], paths)
    
    return len(paths) // 2  


n = int(input("Enter the number of nodes: "))
edges = eval(input("Enter the edges as a list of lists: "))


print(countSpecialNodePaths(n, edges))