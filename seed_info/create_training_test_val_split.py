from random import random

family_file = open('Medium_short_families.txt')
lines = family_file.readlines()

train = .7
test = .2
val = .1

families = []
for line in lines:
    family = line.split(',')[0]
    families.append(family)

train_list = []
test_list = []
val_list = []

for family in families:
    r = random()
    if r < train and len(train_list) < train*(len(families))+1:
        train_list.append(family)
    elif r < train+test and len(test_list) < test*(len(families))+1:
        test_list.append(family)
    elif len(val_list) < val*(len(families))+1:
        val_list.append(family)
    elif len(train_list) < train*(len(families))+1:
        train_list.append(family)
    else:
        test_list.append(family)

print("Train size:", len(train_list))
print("Test size:", len(test_list))
print("Val size:", len(val_list))

train_file = open('Train.txt',mode='w')
test_file = open('Test.txt',mode='w')
val_file = open('Val.txt',mode='w')

train_file.write('\n'.join(train_list) + '\n')
test_file.write('\n'.join(test_list) + '\n')
val_file.write('\n'.join(val_list) + '\n')