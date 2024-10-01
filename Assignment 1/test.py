def fibonaci(n, output):
    a,b=0,1
    while a<n:
        output.append(a)
        a,b = b, a+b 
output = []
fibonaci(1000, output)
print('this is the output, \'you know!\'',output)
# to print \ as part of string, use r before it
print(r'Find is a folder c:\abc\xyz')
# multi string 
print(""" 
Oh Boy!
      This is multi string
And Coming as per 
    space
""")
print("rating is "+ 3*'A')
print('this also'
' works')
word = "Python"
word[:2]   #'Py' character from the beginning to position 2 (excluded)
word[4:]   #'on' characters from position 4 (included) to the end
word[-2:]  #'on' characters from the second-last (included) to the end
word2=word
print(id(word), id(word2)) # will print same 
charList = ['a', 'b', 'c']
charList[1:]=['B','C']
print(charList)# => ['a', 'B', 'C']
print("First Line", "...", "Second Line")



