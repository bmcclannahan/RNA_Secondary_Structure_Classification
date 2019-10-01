def create_rna_family_dictionary(seed_file):
    family_dict = dict()

    current_family = ""
    read_rna = False

    for i in range(len(seed_file)):
        line = seed_file[i]
        if line[0] == '#':
            if line[2:4] == 'GF':
                if line[5:7] == 'AC':
                    current_family = line[10:].rstrip()
                    family_dict[current_family] = []
                elif line[5:7] == 'SQ':
                    read_rna = True
        elif read_rna and len(line) > 10:
            split = line.split(" ")
            rna = split[-1]
            family_dict[current_family].append(rna.rstrip())
        elif line[0:2] == "//":
            read_rna = False
    return family_dict

def print_full_family_information(family_dict):
    for key in family_dict.keys():
        family_array = family_dict[key]
        print(key, "family has", len(family_array),"rna with a gap_included length of", len(family_array[0]))

def print_families_by_length(family_dict):
    short_list = []
    short_count = 0
    medium_list = []
    medium_count = 0
    long_list = []
    long_count = 0
    medium_short_list = []
    for key in family_dict.keys():
        family_length = len(family_dict[key][0])
        if family_length <= 200:
            short_list.append(key)
            short_count += len(family_dict[key])
        if family_length >= 100 and family_length <= 200:
            medium_short_list.append(key)
        if family_length  >= 200 and family_length <= 400:
            medium_list.append(key)
            medium_count += len(family_dict[key])
        if family_length >= 400:
            long_list.append(key)
            long_count += len(family_dict[key])
    print("Short families:",len(short_list),"Medium families:",len(medium_list),"Large families:",len(long_list),"Medium short families:",len(medium_short_list))
    print("Short average:",short_count/len(short_list),"Medium average:",medium_count/len(medium_list),"Long average:",long_count/len(long_list))
    print("\n--------List of Short Families--------")
    for family in short_list:
        print(family, end = ', ')
    print("\n--------List of Medium Families--------")
    for family in medium_list:
        print(family, end = ', ')
    print("\n--------List of Long Families--------")
    for family in long_list:
        print(family, end = ', ')

def sort_families_by_size(family_string):
    return int(family_string[8:])
    
def write_family_lengths_to_files(family_dict):
    short_list = []
    medium_list = []
    long_list = []
    for key in family_dict.keys():
        family_length = len(family_dict[key][0])
        if family_length <= 200:
            short_list.append(str(key)+","+str(len(family_dict[key])))
        if family_length  >= 200 and family_length <= 400:
            medium_list.append(str(key)+","+str(len(family_dict[key])))
        if family_length >= 400:
            long_list.append(str(key)+","+str(len(family_dict[key])))

    short_list.sort(key=sort_families_by_size,reverse=True)
    medium_list.sort(key=sort_families_by_size,reverse=True)
    long_list.sort(key=sort_families_by_size,reverse=True)

    short_file = open('Short_families.txt',mode='w')
    medium_file = open('Medium_families.txt',mode='w')
    long_file = open('Long_families.txt',mode='w')

    short_file.write('\n'.join(short_list) + '\n')
    medium_file.write('\n'.join(medium_list) + '\n')
    long_file.write('\n'.join(long_list) + '\n')

    short_file.close()
    medium_file.close()
    long_file.close()

with open("Rfam.seed") as f:
    seed_file = f.readlines()
print("Seed file has",len(seed_file), "lines")

family_dict = create_rna_family_dictionary(seed_file)
write_family_lengths_to_files(family_dict)
