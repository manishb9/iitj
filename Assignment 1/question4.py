from collections import deque

def getBinaryNumbers(digitToConvert):
    binaryQueue = deque()
    binaryQueue.append('1')
    
    binaryOutput = []
    
    for j in range(digitToConvert):
        current = binaryQueue.popleft()
        binaryOutput.append(current)
        binaryQueue.append(current + '0')
        binaryQueue.append(current + '1')
    
    return binaryOutput

totalNumbersToConver = int(input())

for i in range(totalNumbersToConver):
    digitToConver = int(input())
    result = getBinaryNumbers(digitToConver)
    print(" ".join(result))
