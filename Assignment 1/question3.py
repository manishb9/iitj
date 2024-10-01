def getHeigherIDLowerPopularity(userList,inputLength):
    inputLengthArray = [0] * inputLength
    
    def mergeSort(enum):
        mid = len(enum) // 2
        if mid:
            left, right = mergeSort(enum[:mid]), mergeSort(enum[mid:])
            for i in range(len(enum))[::-1]:
                if not right or left and left[-1][1] > right[-1][1]:
                    inputLengthArray[left[-1][0]] += len(right)
                    enum[i] = left.pop()
                else:
                    enum[i] = right.pop()
        return enum
    
    mergeSort(list(enumerate(userList)))
    return inputLengthArray

inputLength = int(input())
userList = list(map(int, input().split()))

result = getHeigherIDLowerPopularity(userList,inputLength)

print(' '.join(map(str, result)))
