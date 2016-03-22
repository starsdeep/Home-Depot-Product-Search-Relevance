import cPickle

ftest = open('../input/test.csv')

linetest = ftest.readline()
linetest = ftest.readline()
test_product_dict = {}

while linetest:
    termlist = linetest.strip().split(',')
    product_id = termlist[1]
    if not test_product_dict.has_key(product_id):
        test_product_dict[product_id] = 1
    #count += 1
    linetest = ftest.readline()
#    if count > 50:
#        break
#print test_product_dict
cPickle.dump(test_product_dict,open('test_product_dict.pkl','wb'))
    



#line = ftrain.readline()
#while line:
#    print line
#    termlist = line.strip().split(',')
#    print termlist[1]
#    break;
