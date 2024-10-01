userList=[];


def sortArray(leftArray, rightArray):
    i, j =0, 0
    mergedArray = []
    while i<len(leftArray) and j<len(rightArray):
        if(leftArray[i]<rightArray[j]):
            mergedArray.append(leftArray[i])
            i+=1
        else:
            mergedArray.append(rightArray[j])
            j+=1
    if(i<len(leftArray)):
        mergedArray.extend(leftArray[i:])
    if(j<len(rightArray)):
        mergedArray.extend(rightArray[j:])
    return mergedArray

def mergeSort(arr):
    if len(arr)<=1 :
        return arr
    else:
        midPont = len(arr)//2
        leftArray = arr[:midPont]
        rightArray= arr[midPont:]
        lOutPut=mergeSort(leftArray)
        rOutPut=mergeSort(rightArray)
        return sortArray(lOutPut, rOutPut)

def getHeigherIDLowerPopularity(userList):
    enumratedList = list(enumerate(userList))
    mergeSort(enumratedList)
    return outputList


inputLength = int(input())
outputList= [0 for _ in range(inputLength)]    
userInputList = input()
for element in userInputList.split():
    userList.append(int(element))



result = getHeigherIDLowerPopularity(userList)


print(result)
