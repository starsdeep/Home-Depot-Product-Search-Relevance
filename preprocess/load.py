import cPickle

ftrain = open('../input/train.csv')
test_product_dict = cPickle.load(open('test_product_dict.pkl','rb'))
train_product_dict = {}
count = 1

linetrain = ftrain.readline()
linetrain = ftrain.readline()
while linetrain:
    termlist = linetrain.strip().split(',')
    product_id = termlist[1]
    query = termlist[-2].strip('"')
    releavance = float(termlist[-1])
    if test_product_dict.has_key(product_id) and releavance>2:
        if not train_product_dict.has_key(product_id):
            train_product_dict[product_id] = []
            train_product_dict[product_id].append(query)
        else:
            train_product_dict[product_id].append(query)
    count += 1
    linetrain = ftrain.readline()
    if count > 500:
        break

fout = open('product_description_by_query.csv','wb')
fout.write('''"product_uid","product_description"'''+'\r\n')
for key,value in train_product_dict.items():
    fout.write(key+''',"''')
    fout.write(' '.join(value))
    fout.write('''"''')
    fout.write('\r\n')


#print(train_product_dict)


