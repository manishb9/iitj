class BTreeNode:
    def __init__(self, t, leaf):
        self.t = t
        self.leaf = leaf
        self.keys = [None] * (2 * t - 1)
        self.C = [None] * (2 * t)
        self.n = 0

    def traverse(self):
        print(" ", end="")
        for i in range(self.n):
            if not self.leaf:
                self.C[i].traverse()
            print(self.keys[i], end=" ")
        if not self.leaf:
            self.C[self.n].traverse()

    def findTheKey(self, k):
        idx = 0
        while idx < self.n and self.keys[idx] < k:
            idx += 1
        return idx

    def insertNonFull(self, k):
        i = self.n - 1
        if self.leaf:
            while i >= 0 and self.keys[i] > k:
                self.keys[i + 1] = self.keys[i]
                i -= 1
            self.keys[i + 1] = k
            self.n += 1
        else:
            while i >= 0 and self.keys[i] > k:
                i -= 1
            if self.C[i + 1].n == 2 * self.t - 1:
                self.splitTheChild(i + 1, self.C[i + 1])
                if self.keys[i + 1] < k:
                    i += 1
            self.C[i + 1].insertNonFull(k)

    def splitTheChild(self, i, y):
        z = BTreeNode(y.t, y.leaf)
        z.n = self.t - 1
        for j in range(self.t - 1):
            z.keys[j] = y.keys[j + self.t]
        if not y.leaf:
            for j in range(self.t):
                z.C[j] = y.C[j + self.t]
        y.n = self.t - 1
        for j in range(self.n, i, -1):
            self.C[j + 1] = self.C[j]
        self.C[i + 1] = z
        for j in range(self.n - 1, i - 1, -1):
            self.keys[j + 1] = self.keys[j]
        self.keys[i] = y.keys[self.t - 1]
        self.n += 1

    def remove(self, k):
        idx = self.findTheKey(k)
        if idx < self.n and self.keys[idx] == k:
            if self.leaf:
                self.removeFromLeaf(idx)
            else:
                self.removeFromNonLeaf(idx)
        else:
            if self.leaf:
                return
            flag = idx == self.n
            if self.C[idx].n < self.t:
                self.fill(idx)
            if flag and idx > self.n:
                self.C[idx - 1].remove(k)
            else:
                self.C[idx].remove(k)

    def removeFromLeaf(self, idx):
        for i in range(idx + 1, self.n):
            self.keys[i - 1] = self.keys[i]
        self.n -= 1

    def removeFromNonLeaf(self, idx):
        k = self.keys[idx]
        if self.C[idx].n >= self.t:
            pred = self.getPred(idx)
            self.keys[idx] = pred
            self.C[idx].remove(pred)
        elif self.C[idx + 1].n >= self.t:
            succ = self.getSucc(idx)
            self.keys[idx] = succ
            self.C[idx + 1].remove(succ)
        else:
            self.merge(idx)
            self.C[idx].remove(k)

    def getPred(self, idx):
        cur = self.C[idx]
        while not cur.leaf:
            cur = cur.C[cur.n]
        return cur.keys[cur.n - 1]

    def getSucc(self, idx):
        cur = self.C[idx + 1]
        while not cur.leaf:
            cur = cur.C[0]
        return cur.keys[0]

    def fill(self, idx):
        if idx != 0 and self.C[idx - 1].n >= self.t:
            self.borrowFromPrev(idx)
        elif idx != self.n and self.C[idx + 1].n >= self.t:
            self.borrowFromNext(idx)
        else:
            if idx != self.n:
                self.merge(idx)
            else:
                self.merge(idx - 1)

    def borrowFromPrev(self, idx):
        child = self.C[idx]
        sibling = self.C[idx - 1]
        for i in range(child.n - 1, -1, -1):
            child.keys[i + 1] = child.keys[i]
        if not child.leaf:
            for i in range(child.n, -1, -1):
                child.C[i + 1] = child.C[i]
        child.keys[0] = self.keys[idx - 1]
        if not child.leaf:
            child.C[0] = sibling.C[sibling.n]
        self.keys[idx - 1] = sibling.keys[sibling.n - 1]
        child.n += 1
        sibling.n -= 1

    def borrowFromNext(self, idx):
        child = self.C[idx]
        sibling = self.C[idx + 1]
        child.keys[child.n] = self.keys[idx]
        if not child.leaf:
            child.C[child.n + 1] = sibling.C[0]
        self.keys[idx] = sibling.keys[0]
        for i in range(1, sibling.n):
            sibling.keys[i - 1] = sibling.keys[i]
        if not sibling.leaf:
            for i in range(1, sibling.n + 1):
                sibling.C[i - 1] = sibling.C[i]
        child.n += 1
        sibling.n -= 1

    def merge(self, idx):
        child = self.C[idx]
        sibling = self.C[idx + 1]
        child.keys[self.t - 1] = self.keys[idx]
        for i in range(sibling.n):
            child.keys[i + self.t] = sibling.keys[i]
        if not child.leaf:
            for i in range(sibling.n + 1):
                child.C[i + self.t] = sibling.C[i]
        for i in range(idx + 1, self.n):
            self.keys[i - 1] = self.keys[i]
        for i in range(idx + 2, self.n + 1):
            self.C[i - 1] = self.C[i]
        child.n += sibling.n + 1
        self.n -= 1


class BTree:
    def __init__(self, t):
        self.root = None
        self.t = t

    def traverse(self):
        if self.root:
            self.root.traverse()
        else:
            print("None")

    def insert(self, k):
        if not self.root:
            self.root = BTreeNode(self.t, True)
            self.root.keys[0] = k
            self.root.n = 1
        else:
            if self.root.n == 2 * self.t - 1:
                s = BTreeNode(self.t, False)
                s.C[0] = self.root
                s.splitTheChild(0, self.root)
                i = 0
                if s.keys[0] < k:
                    i += 1
                s.C[i].insertNonFull(k)
                self.root = s
            else:
                self.root.insertNonFull(k)

    def remove(self, k):
        if not self.root:
            print("The tree is empty")
            return
        self.root.remove(k)
        if self.root.n == 0:
            if self.root.leaf:
                self.root = None
            else:
                self.root = self.root.C[0]


def parseArray(string):
    string = string.strip()[1:-1]
    return [int(num) for num in string.split(",") if num.strip()]


def main():
    t = int(input())
    inputA = input().strip()
    inputB = input().strip()

    A = parseArray(inputA)
    B = parseArray(inputB)

    tree = BTree(t)

    for i in A:
        tree.insert(i)

    for i in B:
        tree.remove(i)

    if tree.root is None:
        print("None")
    else:
        tree.traverse()
        print()


if __name__ == "__main__":
    main()