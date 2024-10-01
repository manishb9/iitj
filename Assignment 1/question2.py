def getUpdatedSwapedList(lengthOfInput, userList):
    mLinkedList = userList
    
#    for i in range(2, lengthOfInput // 2 + 1, 2):  
    for i in range(2, int(lengthOfInput/2)+1, 2):  
        
        mLinkedList[i - 1], mLinkedList[lengthOfInput - i] = mLinkedList[lengthOfInput - i], mLinkedList[i - 1]
    
    print(" ".join(map(str, mLinkedList)))


lengthOfInput = int(input())
userList = list(map(int, input().split()))
getUpdatedSwapedList(lengthOfInput, userList)
