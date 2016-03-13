#[Home Depot Product Search Relevance](https://www.kaggle.com/c/home-depot-product-search-relevance) on Kaggle

### description


### program environment

* test
* python 3.5
* python library:

    in the requirements.txt
    to install it using : `pip3.5 install -r requirements.txt`
    
### how to run
1. put files into input folder including:attributes.csv,sample_submission.csv,product_descriptions.csv, test.csv, train.csv 
2. run `python3.5 framework.py 1000 ./output/rfr_all`, 1000 means take 1000 instances as train data, you can change it to another number, and -1 means take all instances



### idea

chenqiang

* stem with pos tag using nltk.stem.wordnet.WordNetLemmatizer, not plan yet
* train three classification model to predict 1,2 and 3, output average score of these three as result, not plan yet
* 加特征，title中时否有几个er结尾的单词，query是否有er结尾的单词，query中第一个er结尾的单词出现在title中那些er结尾单词列表的第几个，如果没有出现则为-1, doing


### bad case

* query是树，product是修理树的工具, query是mower，product是mower的布
* query是电池，product是一个工具，同时title中有说不含电池


### performance

| date       | offline |          | online  |   compare  |feature                  | model                   | other trick                                   | comments |
| ---------- |-------- | ---------|---------|------------|-------------------------|-------------------------|-----------------------------------------------|----------|
| 2016-03-03 | 0.47447 |  0       | 0.47571 |    0       |  query_in_title etc     | RandomForestRegressor   | remove stop words                             | base line|
| 2016-03-11 | 0.47477 |  +0.0003 | 0.47587 |   +.00016  |  query_in_title etc     | RandomForestRegressor   | add binary to True in TfidfVectorizer         | base line|
| 2016-03-12 | 0.47335 |  -.00112 | 0.47566 |   -.00005  |                         |                         | {'rfr__max_features': 5, 'rfr__max_depth': 30}|          |

    
