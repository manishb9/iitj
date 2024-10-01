def mergeSort(leactureIntervals):
    if len(leactureIntervals) > 1:
        mid = len(leactureIntervals) // 2
        leftHalf = leactureIntervals[:mid]
        rightHalf = leactureIntervals[mid:]

        mergeSort(leftHalf)
        mergeSort(rightHalf)

        i = j = k = 0
        while i < len(leftHalf) and j < len(rightHalf):
            if leftHalf[i][1] <= rightHalf[j][1]:
                leactureIntervals[k] = leftHalf[i]
                i += 1
            else:
                leactureIntervals[k] = rightHalf[j]
                j += 1
            k += 1

        while i < len(leftHalf):
            leactureIntervals[k] = leftHalf[i]
            i += 1
            k += 1

        while j < len(rightHalf):
            leactureIntervals[k] = rightHalf[j]
            j += 1
            k += 1

def minimumLecturesToSkip(lectures):
    mergeSort(lectures)
    
    lastEndTime = -float('inf')
    skipCount = 0

    for start, end in lectures:
        if start >= lastEndTime:
            lastEndTime = end
        else:
            skipCount += 1

    return skipCount

n = int(input())
lectures = []
for k in range(n):
    start, end = map(int, input().split())
    lectures.append((start, end))

print(minimumLecturesToSkip(lectures))
