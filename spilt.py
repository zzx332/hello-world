import numpy as np
# with  open(r'C:/Users/周喆昕/Desktop/FlyingChairs_release.list' ) as f:
#     print (f.read())
f = open(r'C:/Users/周喆昕/Desktop/FlyingChairs_release.list')
label = open(r'C:/Users/周喆昕/Desktop/FlyingChairs_release_test_train_split.list')
x = open(r'C:/Users/周喆昕/Desktop/train.list', 'w')
c = open(r'C:/Users/周喆昕/Desktop/test.list', 'w')
# print(f.read())
# print(label.read())
train = []
test = []
flines = f.readlines()
llines = label.readlines()
# flines = np.array(flines)
# llines = np.array(llines)
for i in range(len(flines)):
    # print(flines[i])
    # print(llines[i])
    if (llines[i] =='1\n'):
        train.append(flines[i])
    else:
        test.append(flines[i])
x.writelines(train)
c.writelines(test)
f.close()
x.close()
c.close()
label.close()
# print(train)
# f.close()
# label.close()
# print(len(train))
# print(test)
# print(test)
# print(llines[200])
