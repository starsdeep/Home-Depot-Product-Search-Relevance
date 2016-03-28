#Home Depot Product Search Relevance Competition

[Home Depot Product Search Relevance](https://www.kaggle.com/c/home-depot-product-search-relevance) on Kaggle

## 组织形式与目标

### 目标

参加比赛获得好名次是我们共同的目标，但是我更希望的是能在比赛的过程中，真正学到有用的知识。参加这个比赛的现在已经有6个人了，每个人都有自己的分工，或者是收集资料，查看bad case，编写代码，如果每个人能把自己在完成工作的时候遇到的好的资料或者任何有收获的点分享出来，每一个人就相当于做了6份工作，对大家都是一个很好的事情。


### 文档管理
* 关于sklearn pipeline的使用， [这里有一篇很好的文章](http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html)

#### Idea

我们在完成自己任务的过程中，可能突然会冒出很多想法，这些都值得记录下来。每个人看到一个idea觉得make sense, 值得一试，就可以主动的去把这个idea实现,准备实现的时候，在idea那栏里，做个简单的说明，比如“正在实现--廖翊康”，这样大家就不会做到重复的工作。

#### 实验结果
每当测试完一组算法或者feature,希望大家把结果都记录到**实验结果**


#### bad case
在跑实验的时候，查看bad case 是一个很重要的过程，关于bad case的分析，就记录到 **bad case**一栏。

#### 知识分享

祝博，王博可能会看一下这方面的论文，论文的总结可以放到doc文件夹下。其他人比赛的过程中的遇到的任何值得分享的收获也可以放上来。


## 环境配置


 
### program environment

* python 3.5.1 (python 3.4.3 works now as well)
* python library & nltk packges:

    in the requirements.md
    to install pip packages use : `pip3.5 install -r pip_packages.txt`
    
### how to run
1. put files into input folder including : train.csv, test.csv, attributes.csv, product_descriptions.csv google_spell_check_dict.json(typo fixing dict revised from https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos/notebook)
2. write configurations 'config.json' in your directory for experiments. You may copy / refer to other well written configs. For 'num_train' 1000 means take 1000 instances as train data, you can change it to another number, and -1 means take all instances (may take a long time)
3. run `python3.5 framework.py __your_directory__` (e.g. to run baseline, `python3.5 framework.py ./output/rfr_all`).


## Idea

chenqiang


* stem with pos tag using nltk.stem.wordnet.WordNetLemmatizer. [done, chenqinag]
* train three classification model to predict 1,2 and 3, output average score of these three as result. [done chenqiang, not recommended, liaoyikang]
* 加特征, title中时否有几个er结尾的单词，query是否有er结尾的单词，query中第一个er结尾的单词出现在title中那些er结尾单词列表的第几个，如果没有出现则为-1 [done, chenqiang]
* 加特征, 加query和title,description,BM25相似度 [chen qiang done]
* add query features,  该query是否之前出现过, 该query的历史平均relevance, 考虑如果query没有出现如何取平均relevance(平均值，或者其它)
* 观察数据，web 平台， http://139.129.12.69:8088/index.php, search case中relevance为0的数据来自于test.csv,非0的数据来自于train.csv, badcase观察 [chenqiang doing]

fenixlin

* train 13 xgboost classifier to classify 'y>12? y>11? ...' 13 times, predict with probablity sum [doing, gaoben]
* introduce stop words and some rules to cut search query. [done, fenixlin]
* use stanford parser to do semantic analysis and build more features [done, fenixlin]
* use google service to check and fix spelling of search query, see scripts in forum [ done, liaoyikang ]
* 用加权和作为特征来表达共有词匹配的位置信息（越后面的匹配越重要），标题和搜索词都可以做 [ doing, fenixlin]
* 预处理，判断搜索词中每个词在所有搜索词中出现的频繁程度(目前tfidf应该只是行内的tfidf?) [ not planned yet ]
* 将数字和尺寸从标题中单独拿出来作为“副标题”，将型号从标题中单独拿出来作为“副标题” [ not planned yet ]

liaoyikang

* 改进feature 计算过程 缩短每次流程运行的时间[doing, liaoyikang]
* 使用"桶"，将连续值离散化，测试效果[doing, liaoyikang]
* 增加2-gram的feature，见bad case 4[doing, zhuyu]
* 增加feature，是否是连续匹配。比如hot dog，虽然是匹配但不是连续匹配. 见bad case 4 [ not planned yet ]
* 增加feature, 统计词性匹配，有一些形容词匹配，但主题匹配的并没有用，比如query 是 doubl side stapl， title是Double-Sided Organizer，或者见bad case 3. [ done, fenixlin ]


beidouwang

* 测试SVM和SVR，以及各种不同的kernel函数的使用[doing,beidouwang]
* sampling解决skew问题(训练数据中高分数据偏多)，可参考Oversampling（重复sample negative feedback直到和数量和postive相等），undersampling（sample和negative feedback一样多的postive feedback），另外专门有论文讲cost sensitive sampling [ doing, zhuyu ]
* 使用scikit-learn的feature selection模块减少无用feature [ not planned yet ]


## Bad case 分析

* query是树，product是修理树的工具, query是mower，product是mower的布


###典型bad case

注意有些badcase就是数据里面的噪音，就是标错了，可能同样的情况很多sample是３分就它是１分，不必强行分析

|编号| query | title |原因|改进方法|解决情况|
|---|---|---|---|---|---|
| 1 |Ryobi ONE+ 18 in. 18-Volt Lithium-Ion Cordless Hedge Trimmer - Battery and Charger Not Included|18volt. batteri charger|主语不一样，反义处理 not included 处理|标题主体提取|已解决|
|2|topiari tree|Romano 4 ft. Boxwood Spiral Topiary Tree|query是简单的词，title却是一个很详细的东西，所以应该是一般性匹配|增加query len / title/len||
|3|bronz green|Green Matters 3-Light Mahogany Bronze Vanity Fixture|主题词不匹配，green 形容词匹配，没啥用|名词匹配|已解决|
|4|hot dog|HealthSmart Digger Dog Reusable Hot and Cold Pack||2-gram||


## 实验结果记录

| submit date | name | offline |          | online  |   compare  |feature                  | model                   | other trick                                   | comments |
| ---------- |-------- | --------|---------|---------|------------|-------------------------|-------------------------|-----------------------------------------------|----------|
| 2016-03-03  | chenqiang | 0.47447 |  0       | 0.47571 |    0       |  query_in_title etc     | RandomForestRegressor   | remove stop words                             | base line|
| 2016-03-11  | chenqiang | 0.47477 |  +0.0003 | 0.47587 |   +.00016  |  query_in_title etc     | RandomForestRegressor   | add binary to True in TfidfVectorizer         |          |
| 2016-03-12  | chenqiang | 0.47335 |  -.00112 | 0.47566 |   -.00005  |                         |                         | {'rfr__max_features': 5, 'rfr__max_depth': 30}|          |
| unsubmitted | fenixlin  | 0.49002 |          |          |            |                         | XGBoost Regressor      |                                               |          |
| unsubmitted | fenixlin  | 4.83365 |          |          |            |                         | RandomForestClassifier | classify once with independent labels(1~13)   |          |
| 2016-03-16  | chenqiang | 0.47284 |  -.00163 |  0.47566 |   -.00005  |  fixed bug replacing query_in_description with query_in_title | RandomForestRegressor | stem with pos tag using nltk.stem.wordnet.WordNetLemmatizer|
| 2016-03-17  | chenqiang | 0.47083 |  -.00364 |  0.47824 |   +.00253  |  {'rfr__max_features': 7, 'rfr__max_depth': 50}  加入特征，, "title_query_BM25", "description_query_BM25"|  RandomForestRegressor|                |          |
| 2016-03-17  | chenqiang | 0.47046 |  -.00401 |  0.47677 |   +.00106  |  {'rfr__max_features': 5, 'rfr__max_depth': 50}  加入特征，, er        |  RandomForestRegressor|                |          |
| 2016-03-18  | fenixlin  | 0.47249 |  -.00198 |  0.47484 |   -.00087  |  rfr_all{5,30}+删除无主体的搜索词  　  |  RandomForestRegressor|                |          |
| 2016-03-19  | chenqiang | 0.6, 0.6, 0.6|     |  0.50780 |   +.3209   | BM25, no er, get optimised depth and features|  train three classification model |   |                 |
| 2016-03-19  | liaoyikang |0.47265|     |   |      | add search_term_clean 做了拼写纠错好去除stopwords| rfr |   |                 |
| 2016-03-21  | fenixlin  | 0.46908 |  -.00569 |  0.46966 |   -.00605  |  rfr {5, 30}+postag统计标题+search_term_clean  　  |  RandomForestRegressor|                |          |
| 2016-03-24  | fenixlin  | 0.46297 |  -.01180 |  0.46366 |   -.01205  |  rfr {10, 30}+标题主题提取，详见config  　  |  RandomForestRegressor|                | Runtime: 1.5h |
| 2016-03-25  | fenixlin  | 0.46195 |  -.01282 |  0.46188 |   -.01383  |  rfr {2000, 12, 35}+ratio/按序匹配特征，详见config  　  |  RandomForestRegressor|                | Runtime: 1.5h |
| 2016-03-26  | chenqiang | 0.32271 |  -.15176 |  0.51258 |   +.03687  |  rfr {2000, 12, 35} add ,"query_count_in_train" ,"query_relevance_vg_in_train" base last submission  　  |  RandomForestRegressor|                |  over fitting absolutely |
| 2016-03-26  | chenqiang | 0.46305 |  -.01142 |  0.51258 |   +.03687  |  rfr {2000, 12, 35} add ,"query_count_in_train" base last submission  　  |  RandomForestRegressor|                |  over fitting absolutely |
| 2016-03-27  | fenixlin  | 0.45982 |  -.01465 |  0.45953 |   -.01618  |  rfr {2000, 12, 38}+标题主题提取改进，详见config  　  |  RandomForestRegressor|                | Runtime: 1.5h |
| 2016-03-28  | liaoyikang  | 0.45978 |  -0.01468 |  0.45946 |   -0.01625  |  rfr {2000, 12, 38}+bigram_feature，详见config  　  |  RandomForestRegressor|                | Runtime: 10min |


