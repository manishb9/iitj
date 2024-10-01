arr = [6,4,2,1,9,8,3,5]
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
print(mergeSort(arr))