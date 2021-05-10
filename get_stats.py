import numpy as np
from collections import Counter
import sys
#with open('./temp/run_en_de/test/test.out') as f: #.format(lang)) as f:
#    p = f.readlines()
#with open('output.out.decoded') as f:
#    q = f.readlines()
#
#
#with open('./temp/run_en_de/data/test.src') as f: #.format(lang)) as f:
#    a = f.readlines()
#with open('output.txt.encoded') as f:
#    b = f.readlines()

a = open(sys.argv[1]).readlines()
b = open(sys.argv[2]).readlines()
p = open(sys.argv[3]).readlines()
q = open(sys.argv[4]).readlines()

#tmp = [(i, j)    for i, j in zip(a, p)    if len(i.split()) <= 40] ## not needed 
#a, p = zip(*tmp)
a = a[:len(b)]
p = p[:len(q)]

a,b,p,q = list(map(np.array,[a,b,p,q]))


## Do not take set intersection since q can have repeating entries like translation for common phrases like "thank you" etc.
print("Correct:",np.sum(p==q))
print("Total:",len(q))

print("Success Rate:",np.sum(p==q)/len(q))

'''src = a[p==q]
adv_src = b[p==q]

new_src = []
new_adv_src = []
## Do something for unequal lengths
for i in range(len(adv_src)):
    if len(src[i].split())!=len(adv_src[i].split()):
        print("Unequal length for index",i)
    else:
        new_src.append(src[i])
        new_adv_src.append(adv_src[i])

dist = lambda x: np.sum(np.array(x[0].split())!=np.array(x[1].split())) / len(x[0].split())
#l = list(map(dist,zip(src,adv_src)))
l = list(map(dist,zip(new_src,new_adv_src)))
print(np.mean(l),np.median(l),np.max(l),np.min(l))'''

dist = lambda x: np.sum(np.array(x[0].split())!=np.array(x[1].split())) / len(x[0].split())
l = list(map(dist,zip(a,b)))
print(np.mean(l),np.median(l),np.max(l),np.min(l),len(l))
