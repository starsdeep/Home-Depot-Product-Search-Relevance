__author__ = 'fucus'
import pandas as pd
import config.project as project

def process_product(limit=-1):
    counter = 0
    df_pro_desc = pd.read_csv(project.original_product_descriptions, encoding="ISO-8859-1")
    product_insert_file = open("product_insert.sql", "w")
    for index, item in df_pro_desc.iterrows():
        desc = str(item["product_description"]).replace('''"''', '''\\"''')
        uid = item["product_uid"]
        insert_line = """INSERT INTO product(product_uid, product_description) VALUE ("%s", "%s"); \n""" \
                      % (uid, desc)
        product_insert_file.write(insert_line)
        counter += 1
        if limit > 0 and counter > limit:
            break


def process_title(limit=-1):
    counter = 0
    df_train = pd.read_csv(project.original_train_file, encoding="ISO-8859-1")
    df_test = pd.read_csv(project.original_test_file, encoding="ISO-8859-1")
    product_title_set_file = open("product_title_set.sql", "w")
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True)

    for index, item in df.iterrows():
            title = str(item["product_title"]).replace('''"''', '''\\"''')
            uid = item["product_uid"]
            insert_line = """UPDATE product SET product_title = "%s" WHERE product_uid = "%s" ; \n""" \
                          % (title, uid)
            product_title_set_file.write(insert_line)
            counter += 1
            if limit > 0 and counter > limit:
                break

def process_attr(limit=-1):
    counter = 0
    df_attr = pd.read_csv(project.original_attr_file, encoding="ISO-8859-1")
    product_attr_insert = open("product_attr_insert.sql", "w", encoding="utf8")
    for index, item in df_attr.iterrows():
        try:
            uid = int(item["product_uid"])
        except:
            continue
        name = str(item["name"]).replace('''"''', '''\\"''')
        value = str(item["value"]).replace('''"''', '''\\"''')
        insert_line = """INSERT INTO product_attribute(product_uid, `name`, `value`) VALUE ("%s", "%s", "%s"); \n""" \
                      % (uid, name, value)
        product_attr_insert.write(insert_line)
        counter += 1
        if limit > 0 and counter > limit:
            break
def process_search_case(limit=-1):
    counter = 0
    df_train = pd.read_csv(project.original_train_file, encoding="ISO-8859-1")
    df_test = pd.read_csv(project.original_test_file, encoding="ISO-8859-1")
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True)

    case_file = open("product_search_case_insert.sql", "w", encoding="utf8")
    for index, item in df.iterrows():
        try:
            case_id = int(item["id"])
            uid = int(item["product_uid"])
        except:
            continue

        try:
            relevance = float(item["relevance"])
        except:
            relevance = 0

        query = str(item["search_term"]).replace('''"''', '''\\"''')
        title = str(item["product_title"]).replace('''"''', '''\\"''')
        insert_line = """INSERT INTO search_case(case_id, product_uid, relevance, query, product_title) VALUE ("%s", "%s", "%s", "%s", "%s"); \n""" \
                      % (case_id, uid, relevance, query, title)
        case_file.write(insert_line)
        counter += 1
        if limit > 0 and counter > limit:
            break

if __name__ == '__main__':
    process_search_case()