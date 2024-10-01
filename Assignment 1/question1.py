from collections import Counter

def getElementsCountToRemove(lengthOfInput, userArray):
    frequencyNumber = Counter(userArray)
    
    sortedFrequency = sorted(frequencyNumber.values(), reverse=True)
    
    removedCount = 0
    currentSize = lengthOfInput
    

    for f in sortedFrequency:
        currentSize -= f
        removedCount += 1
        if currentSize <= lengthOfInput // 2:
            break
    
    return removedCount

lengthOfInput = int(input())
userArray = list(map(int, input().split()))

print(getElementsCountToRemove(lengthOfInput, userArray))