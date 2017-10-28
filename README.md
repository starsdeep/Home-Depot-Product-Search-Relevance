# Top 1% Solution Home Depot Product Search Relevance Competition

[Home Depot Product Search Relevance](https://www.kaggle.com/c/home-depot-product-search-relevance) on Kaggle

Credit to @fenixlin and @bengxy

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
* 用加权和作为特征来表达共有词匹配的位置信息（越后面的匹配越重要），标题和搜索词都可以做 [ done, fenixlin, useless]
* 预处理，判断搜索词中每个词在所有搜索词中出现的频繁程度(目前tfidf应该只是行内的tfidf?) [ done, fenixlin, liaoyikang ]
* 将数字和尺寸从标题中单独拿出来作为“副标题”，将型号从标题中单独拿出来作为“副标题” [ done, fenixlin, useless ]
* 考虑搜索词中的型号信息，从未stem分隔数字和字母的原搜索词中提取含有数字的部分 [ done, fenixlin ]
* idf特征增加：搜索词用idf和来归一化、排除数字后的idf匹配 [doing, fenixlin]
* 匹配特征补充：匹配词相隔的最远距离、带距离限制的顺序匹配、逆序匹配、search term的last noun匹配、search term的noun匹配覆盖率 [doing, fenixlin]
* 尝试增加tfidf-svd分解的维度 [ done, fenixlin, useless ]
* 尝试使用TfidfVectorizer的binary参数　[not planned yet]
* 考虑fuzzy match的更广泛应用 [not planned yet]
* 考虑word2vec的应用 [not planned yet]
* 考虑pos_tag结合idf产生新的特征 [ done, fenixlin, useless ]
* 处理numsize的额外的空格, 考虑为numsize采用不同的匹配方式(str.find()模糊匹配应该出现在3匹配3in.，但不能匹配1/3in.) [not planned yet]
* 利用更多的attributes(如bullet01~09) [not planned yet]
* 尝试更多模型：LR(预期0.47) [not planned yet]

- 以下idea来源于Crowdflower Search Results Relevance比赛 [not planned yet]
* n-gram / unique n-gram的计数(unigram,bigram,trigram都当作搜索语素) [not planned yet]
* 匹配position特征：最左最右和中位数平均值标准差-normalize, 计算cos_sim [not planned yet]
* 尝试改变distance: Jaccard & Dice  [not planned yet]
* 每个(r), (q, r)与当前样本相同的组内样本距离的统计值（左中右中均方） [not planned yet]
* 拓展tfidf到2gram, 3gram, 计算q,t,d相互之间的cos_sim [not planned yet]
* 产生queryId(LabelBinarizer)(注意参考的比赛本身是没有pid而有qid的) [not planned yet]
* 计算term之间unigram/bigram的cooccurence的tfidf [not planned yet]
* Query expansion: (所有/高分的)title拼到query上取最频繁的词(10~15 top，然后再做匹配特征)(注意有效可能是因为该比赛只有261个unique query)(第三名分析高分占大部分所以用了所有title) [not planned yet]
* query和title之间的lzma距离(difflib-SequenceMatcher), mean/max编辑距离 [not planned yet]
* last word of query [not planned yet]
* letter frequency: (FreqQ+A)/(FreqT+B)/(Qlen) [not planned yet]
* BoW+Tfidf->SVD->tSNE? (比赛里面SVD后竟然都有200~250维) [not planned yet]
* 第三名也有用到round to integer的方法，不过也有可能是因为他们的评价标准是kappa score [not planned yet]
* 更好的main_title_extraction(https://github.com/skrypka/crowdflower-search/blob/master/extractProductName_NEW.R) [not planned yet]
* intra-title simliarity: 同一个query，本sample的title和其他各个title(按打分划分组别来求max/mean)之间的相似度？见第三名答案 [not planned yet]
* antonym: 手工构建的反义词区分feature，比如men shoes - women shoes, 构建men - women说明需求的差异 [not planned yet]
* query不同位置的词匹配title的信息(比如7th就分了第一、第二、中间、最后) + 匹配title不同位置的信息(比如第1/2匹配第1/2，最后匹配最后) [not planned yet]
* titleRatio / descRatio，匹配后除以title和description的长度 [not planned yet]


liaoyikang

* 改进feature 计算过程 缩短每次流程运行的时间[doing, liaoyikang]
* 使用"桶"，将连续值离散化，测试效果[doing, liaoyikang]
* 增加2-gram的feature，见bad case 4[doing, zhuyu]
* 增加feature，是否是连续匹配。比如hot dog，虽然是匹配但不是连续匹配. 见bad case 4 [ not planned yet ]
* 增加feature, 统计词性匹配，有一些形容词匹配，但主题匹配的并没有用，比如query 是 doubl side stapl， title是Double-Sided Organizer，或者见bad case 3. [ done, fenixlin ]


beidouwang

* 测试SVM和SVR，以及各种不同的kernel函数的使用[doing,beidouwang]
* sampling解决skew问题(训练数据中高分数据偏多)，可参考Oversampling（重复sample negative feedback直到和数量和postive相等），undersampling（sample和negative feedback一样多的postive feedback），另外专门有论文讲cost sensitive sampling [ done, zhuyu ]
* 使用scikit-learn的feature selection模块减少无用feature [ not planned yet ]


## Bad case 分析

* query是树，product是修理树的工具, query是mower，product是mower的布


### 典型bad case

注意有些badcase就是数据里面的噪音，就是标错了，可能同样的情况很多sample是３分就它是１分，不必强行分析

|编号| query | title |原因|改进方法|解决情况|
|---|---|---|---|---|---|
| 1 |Ryobi ONE+ 18 in. 18-Volt Lithium-Ion Cordless Hedge Trimmer - Battery and Charger Not Included|18volt. batteri charger|主语不一样，反义处理 not included 处理|标题主体提取|已解决|
|2|topiari tree|Romano 4 ft. Boxwood Spiral Topiary Tree|query是简单的词，title却是一个很详细的东西，所以应该是一般性匹配|增加query len / title/len||
|3|bronz green|Green Matters 3-Light Mahogany Bronze Vanity Fixture|主题词不匹配，green 形容词匹配，没啥用|名词匹配|已解决|
|4|hot dog|HealthSmart Digger Dog Reusable Hot and Cold Pack||2-gram||


## need to check
* utility.num_common_word_ordered
* utility.num_size_word
* utility.numsize_of_str


## feature 含义

### from FirstFeatureFuncDict

* origin_search_term, 最原始的search_term
* ori_stem_search_term, 对原始的search_term,通过utility.str_stem(s, by_lemmatizer=False)处理过的结果
* search_term,对原始的search_term，通过feature.search_term_clean，处理过的结果，主要是纠错和删除stopword，同时还会调用utility.str_stem
* main_title，对原始的product_title，进行utility.main_title_extract，处理之后，再进行utility.str_stem处理
* title,对原始的product_title，进行utility.str_stem处理的结果
* description,对原始的product_description进行utility.str_stem处理的结果
* brand,原始brand，来自于attributes.csv的MFG Brand Name字段的值，对原始brand进行utility.str_stem处理之后的结果


* ori_query_in_title, ori_stem_search_term整体在title中出现的次数
* query_in_main_title, search_term(非原始search_term)整体在main_title中出现的次数
* query_in_title, search_term(非原始search_term)整体在title中出现的次数
* query_in_description, search_term(非原始search_term)整体在description中出现的次数 
* query_last_word_in_main_title, search_term(非原始search_term)的最后一个单词在是否在main_title中，【模糊匹配算0.5】
* query_last_word_in_title,search_term(非原始search_term)的最后一个单词在是否在title中，【模糊匹配算0.5】
* query_last_word_in_description,search_term(非原始search_term)的最后一个单词在是否在description中，【模糊匹配算0.5】
* word_in_main_title,search_term(非原始search_term)中所有词，有多少个在main_title中出现了，【模糊匹配算0.5】 
* word_in_main_title_ordered, 对search_term(非原始search_term)进行split,从第一个词开始顺序匹配main_title，看能匹配多少个词，【这里可以改进，因此如果第一个词出现在main_title最后面的话，返回值就是1，这会忽略search_term其它词的作用】
* word_in_title,search_term(非原始search_term)中所有词，有多少个在title中出现了，【模糊匹配算0.5】




* ori_word_in_title_ordered, 对ori_stem_search_term进行split,从第一个词开始顺序匹配，看能匹配多少个词，【这里可以改进，因此如果第一个词出现在main_title最后面的话，返回值就是1，这会忽略search_term其它词的作用】
* word_in_title_ordered, 对search_term(非原始search_term)进行split,从第一个词开始顺序匹配title，看能匹配多少个词，【这里可以改进，因此如果第一个词出现在main_title最后面的话，返回值就是1，这会忽略search_term其它词的作用】
* word_in_description,search_term(非原始search_term)的各个词语，有多少个出现description中
* word_in_brand,search_term(非原始search_term)的各个词语，有多少个出现brand中


* bigram_in_title,???对search_term切分的时候，所有相邻的两个单词组成一个bigram word，返回list of  bigram word， 这些bigram word有多少个出现在title中
* bigram_in_main_title,???对search_term切分的时候，所有相邻的两个单词组成一个bigram word，返回list of  bigram word， 这些bigram word有多少个出现在main_title中
* bigram_in_description,???对search_term切分的时候，所有相邻的两个单词组成一个bigram word，返回list of  bigram word， 这些bigram word有多少个出现在description中
* bigram_in_brand,???对search_term切分的时候，所有相邻的两个单词组成一个bigram word，返回list of  bigram word， 这些bigram word有多少个出现在brand中


* search_term_fuzzy_match, search_term, title????
* len_of_search_term_fuzzy_match, search_term_fuzzy_match中的单词数量


* word_with_er_count_in_query, search_term中，er结尾的单词的数目
* word_with_er_count_in_title,  title中，er结尾的单词的数目
* first_er_in_query_occur_position_in_title, search_term中第一个er结尾的单词出现在title中的位置

* len_of_query,search_term(非原始search_term)中单词的个数
* len_of_main_title,main_title中单词的个数
* len_of_title,title中单词的个数
* len_of_description,description中单词的个数
* len_of_brand,brand中单词的个数


* chars_of_query, search_term(非原始search_term)的字符个数，包含空格字符


* ratio_main_title, word_in_main_title / ( len_of_query + 1 )
* ratio_title, word_in_title / ( len_of_query + 1 )
* ratio_main_title_ordered, word_in_main_title_ordered / (len_of_query + 1)
* ratio_title_ordered, word_in_title_ordered / (len_of_query + 1)
* ratio_description, word_in_description / ( len_of_query + 1)
* ratio_brand, word_in_brand / (len_of_query + 1)



* ratio_bigram_title, bigram_in_title / (len_of_query + 1)
* ratio_bigram_main_title, bigram_in_main_title / (len_of_query + 1)
* ratio_bigram_description, bigram_in_description / (len_of_query + 1)
* ratio_bigram_brand, bigram_in_brand / (len_of_query + 1)


* title_query_BM25， title和search_term之间的BM25相似度，离线计算???
* description_query_BM25，description 和search_term之间的BM25相似度，离线计算？？


### from PostagFeatureFuncDict，之后的search_term都是指原始search_term经过utility.str_stem之后的内容

* noun_of_query, search_term中名词的个数
* noun_of_title, title 中名词的个数
* noun_of_main_title, main_title中名词的个数
* noun_of_description, description中名词的个数
* noun_match_main_title, search_term中的词语在main_title中，并且是名词（在main_title中是名词）的个数，有模糊匹配的词算0.5个
* noun_match_title, search_term中的词语在title中，并且是名词（在title中是名词）的个数，有模糊匹配的词算0.5个
* noun_match_main_title_ordered, 对search_term进行split,从第一个词开始顺序匹配main_title中的名词，看能匹配多少个词，【这里可以改进，因此如果第一个词出现在main_title最后面的话，返回值就是1，这会忽略search_term其它词的作用】
* noun_match_title_ordered, 含义同上，不过把main_title替换成了title
* noun_match_description, 含义同noun_match_main_title_ordered，不过main_title替换成description
* match_last_noun_main, 对于main_title当中的最后一个名词，是否在search_term中可以找到
* match_last_2_noun_main,对于main_title当中的最后两个名词，有多少个可以在search_term中找到
* match_last_3_noun_main,对于main_title当中的最后三个名词，有多少个可以在search_term中找到
* match_last_5_noun_main,对于main_title当中的最后五个名词，有多少个可以在search_term中找到



### from NumsizeFuncDict

* numsize_word_in_main_title, ？？？？ search_term中的度量单位词，包含数字和单位,在main_title中，可以找到多少个匹配的词
* numsize_word_in_title， 同numsize_word_in_main_title，不过要将main_title替换成title
* numsize_word_in_description, 同numsize_word_in_main_title，不过要将main_title替换成description

* len_of_numsize_query, ？？？？search_term中有多少度量单位词
* len_of_numsize_main_title, ??? main_title中有多少度量单位词
* len_of_numsize_title， ？？？title中有多少度量单位词
* len_of_numsize_description， ？？ description有多少度量单位词

* numsize_title_case1,  len(numsize_of_query) == 0 and len(numsize_of_title)==0,
* numsize_title_case2, len(numsize_of_query)==0 and len(numsize_of_title)>0),
* numsize_title_case3,len(numsize_of_query)>0 and len(numsize_of_title)==0),
* numsize_title_case4,len(numsize_of_query)>0 and len(numsize_of_title)>0 and len(set(numsize_of_query) & set(numsize_of_title))==0),
* numsize_title_case5, len(numsize_of_query)>0 and len(numsize_of_title)>0 and len(set(numsize_of_query) & set(numsize_of_title))>0),
* 以上几个case都是，numsize_of_query和numsize_of_title的一些组合

* numsize_description_case1, len(numsize_of_query)==0 and len(numsize_of_description)==0),
* numsize_description_case2,len(numsize_of_query)==0 and len(numsize_of_description)>0),
* numsize_description_case3,len(numsize_of_query)>0 and len(numsize_of_description)==0),
* numsize_description_case4,len(numsize_of_query)>0 and len(numsize_of_description)>0 and len(set(numsize_of_query) & set(numsize_of_description))==0),
* numsize_description_case5, len(numsize_of_query)>0 and len(numsize_of_description)>0 and len(set(numsize_of_query) & set(numsize_of_description))>0),
* 以上几个case都是，numsize_of_query和numsize_of_description的一些组合

### from LastFeatureFuncDict

* ratio_noun_match_title, noun_match_title / ( noun_of_query + 1 )
* ratio_noun_match_main_title, noun_match_main_title / ( noun_of_query + 1 )
* ratio_noun_match_description, noun_match_title / ( noun_of_query + 1 )
* ratio_noun_match_title_ordered, noun_match_title_ordered / ( noun_of_query + 1 )
* ratio_noun_match_main_title_ordered,  noun_match_main_title_ordered / ( noun_of_query + 1 )
* ratio_noun_match_description, noun_match_description / ( noun_of_query + 1 ) 


* ratio_numsize_main_title,  numsize_word_in_main_title / ( len_of_numsize_query + 1 )
* ratio_numsize_title, numsize_word_in_title / ( len_of_numsize_query + 1 )
* ratio_numsize_description, numsize_word_in_description / ( len_of_numsize_query + 1 ) 



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
| 2016-03-29  | liaoyikang  | 0.45960 |  -0.01487 |  0.45937 |   -0.01634  |  rfr {2000, 12, 38}+stem改进+bigram，详见config  　  |  RandomForestRegressor|                | Runtime: 10min |
| 2016-03-29  | zhuyu  | 0.4595 |  -0.01492 |  0.45917 |   -0.01654  |  rfr {2000, 12, 38}+num_size+bigram，详见config  　  |  RandomForestRegressor|                | Runtime: 10min |
| 2016-03-29  | liaoyikang| 0.45957 |  |  0.45941 |   |  rfr {2000, 12, 38} + all numsize feature，详见rfr_liaoyikang/config  　  |  RandomForestRegressor|                | |
| 2016-03-29  | liaoyikang| 0.45958 |  |  0.45921 |   |  rfr {2000, 12, 38} + categorical numsize feature，详见rfr_liaoyikang/config  　  |  RandomForestRegressor|                | |
| 2016-04-02  | chenqiang| 0.46306 |  |  0.46802 |   |  {'rfr__max_depth': 38, 'rfr__max_features': 12, 'rfr__n_estimators': 2400} all features， rfr_chenqiang/2016-04-02.config.json  　  |  RandomForestRegressor|                | |
| 2016-04-03  | chenqiang| 0.46242 |  |  0.46835 |   |  {'rfr__max_depth': 38, 'rfr__max_features': 12, 'rfr__n_estimators': 2400} add query_is_general， rfr_chenqiang/2016-04-03.config.json  　  |  RandomForestRegressor|                | |
| unsubmitted | fenixlin  | 0.46013 |          |          |            | 在num_common_word中默认使用精确匹配而不是str.find这种前缀匹配  | RandomForest Regressor      |                                               |          |
| unsubmitted | fenixlin  | 0.46934 |          |          |            | 在title/description/main title中去掉numsize, typeid  | RandomForest Regressor      |                                               |          |
| 2016-04-03 | fenixlin  | 0.45871 |          | 0.45882 |            | rfr{2000,38,12} + 更精确的typeid特征。顺便测试这一段时间内对原有功能改动的影响。详见config | RandomForest Regressor      |                                               |          |
| 2016-04-03 | fenixlin  | 0.43887 |          | 0.46408 |            | rfr{2000,38,12} + 同一product/search term的历史平均relevance(预处理得到，训练中已去掉自身relevance)详见config | RandomForest Regressor      |                                               | 历史数据可能不靠谱 |
| 2016-04-04 | chenqiang | 0.45802 |           |            |       |  rfr{2000,38,12} + add word_in_concatenated_type feature, see more from rfr_chenqiang/2014-04-04| RandomForest Regressor      |                                               | 添加商品类型的匹配信息 |
| 2016-04-04 | chenqiang | 0.45778 |           |  0.46190   |       |  rfr{2000,38,12} + add query_is_general feature, see more from rfr_chenqiang/2014-04-04-02| RandomForest Regressor            | | |
| 2016-04-05 | liaoyikang | 0.45746 |           |  0.45766  |       |  rfr{2000,38,12} + idf featues, rfr_liaoyikang/config.json        to    | | |
| 2016-04-05 | fenixlin  | 0.45612 |          | 0.45744 |            | rfr{2000,42,12} + 更多idf特征(模糊匹配，search_term idf，search_term匹配)详见config | RandomForest Regressor      |                                               |  |
| unsubmitted | fenixlin  | 0.45623 |          |  |            | rfr{2000,42,12} + idf+postag特征，详见config | RandomForest Regressor      |                                               |  |
| 2016-04-08 | fenixlin  | 0.45455 |          | 0.45381 |            | xgbr{0.01,11,800,3,0.7,0.48} + 提交结果为0.45612的特征，详见config | Xgboost Regressor      |                                               |  |
| 2016-04-10 | chenqiang | 0.45331  |          | 0.45400 |       | add query_is_none_sense feature, query_is_general, add correction ,edsel to edsal, guage to gauge, base on last submit xgbr{0.01,11,800,3,0.7,0.48} + all useful features, see more from xgbr_fenixlin/160408submission01.json | Xgboost Regressor      |                                               |  | 
| 2016-04-13 | zhuyu | 0.454722  |          | 0.45338 |       | modify testing data, extend title by query, see more from xgbr_zhuyu/160413submission01.json | Xgboost Regressor      |                                               |  | 
| 2016-04-14 | fenixlin  | 0.45402 |          | 0.45323 |            | xgbr{0.01,11,800,3,0.7,0.48} + 改善好的匹配统计特征(未做zhuyu学长的title拓展)，详见config | Xgboost Regressor      |                                               |  |
