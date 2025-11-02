I= 34
print(type(I))
print('Integer : ',(I))

F = 5.3
print(type(F))
print('Float:',(F))

S= "Zaid"
print(type(S))
print("string :",(S))

user_input=input('please enter a integer: ')
print('input integer :',(user_input))
print(type(user_input))

Sum= I + F
Sub= I - F
Multiply= I * F
division= I / F
print(type(Sum))
print(type(Sub))
print(type(Multiply))
print(type(division))

print('Sumddition:',(Sum))
print('Subtraction:',(Sub))
print('MultiplicSumtion:',(Multiply))
print('division:', division)

if I > F:
    print('Integer is greater than float')
elif I==F:
    print('Both are equal')
else: 
    print('Integer is less than float')

if I > F or I < F:
    print(' Greater less than Integer')
else:
    print('Both are Equal')

if I > F and I < F:
    print(' Greater less than Integer')
else:
    print('Both are Equal')
St= "Sheikh zaid is the student of nexskills institute. he's taking course of Full stack AI "

print('Nexskills Student:',(St))
print(type(St))
print(St[30:38])
print(St[:11])

for i in St:
        if i == 'a':
            break
        print(i)
for i in St:
        if i == 'a':
            continue
        print(i)
x= 1
while x < 3 :
    StopIteration
    print(x)
    x=x + 1

print(len(St))
print(St[::-1])

