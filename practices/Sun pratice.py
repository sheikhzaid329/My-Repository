books= [23,1210,'novel',True,'st william', 2000]
print(books)
print(type(books))

novel=books[2:3]
print(books[2:3])
print(type(novel))


for i in books:
    print(i)

books.append('publisher')
books.insert(2,'reviewer')

books.remove('reviewer')
books.pop(1)
 
list_books=(23,1210,'novel',True,'st william', 2000)

print(type(list_books))

print(list_books[-4:-3])


