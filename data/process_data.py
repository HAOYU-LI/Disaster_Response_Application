import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge dataset:
    df = messages.merge(categories, on=['id'], how='inner')
    #Split categories into separate category columns.
    categories = pd.concat([categories['id'], categories['categories'].\
                            str.split(";", expand = True)], axis = 1)
    #rename columns:
    row = categories.iloc[0,1:]
    category_colnames = ['id'] + list(map(lambda x : x[0:-2], list(row)))
    categories.columns = category_colnames

    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1] if type(x) == str else x)
        
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x : int(x))

    #Replace categories column in df with new category columns:
    df.drop(labels = ['categories'], axis=1, inplace=True)
    df = df.merge(categories, on='id', how='inner')
    return df


def check_duplicate(df):
    count, dup_ids = 0, []
    row, col = df.shape
    seen = set()
    for i in range(row):
        if tuple(df.loc[i]) in seen:
            count+=1
            dup_ids.append(i)
        else:
            seen.add(tuple(df.loc[i]))
    return dup_ids


def clean_data(df):
    #remove duplicate rows:
    duplicate_index = check_duplicate(df)
    df.drop(labels=duplicate_index, axis=0, inplace=True)
    df = df.reindex(range(df.shape[0]))
    return df



def save_data(df, database_filename):
    engine = create_engine('sqlite:///%s'%database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()