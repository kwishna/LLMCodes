
# !pip install -qU pinecone-client openai datasets
import openai
import os
import pinecone
import os
import pandas as pd
import numpy as np
import time
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from statistics import mean

openai.api_key = "<<YOUR SECRET KEY>>"
MODEL = "text-embedding-ada-002"

res = openai.Embedding.create(input=[
  "Sample document text goes here",
  "there will be several phrases in each batch"
],
engine=MODEL)

print(res)

print(
  f"vector 0: {len(res['data'][0]['embedding'])}\nvector 1: {len(res['data'][1]['embedding'])}"
)

# we can extract embeddings to a list
embeds = [record['embedding'] for record in res['data']]
len(embeds)

len(embeds[0])

# ----------

index_name = 'semantic-search-openai'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
  api_key="PINECONE_API_KEY",
  environment="YOUR_ENV"  # find next to api key in console
)

# check if 'openai' index already exists (only create index if not)
if index_name not in pinecone.list_indexes():
  pinecone.create_index(index_name, dimension=len(embeds[0]))

# connect to index
index = pinecone.Index(index_name)

# -------

from datasets import load_dataset

# load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:1000]')
print(trec[0])

# ----------

from tqdm.auto import tqdm

count = 0  # we'll use the count to create unique IDs
batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(trec['text']), batch_size)):
  # set end position of batch
  i_end = min(i + batch_size, len(trec['text']))
  # get batch of lines and IDs
  lines_batch = trec['text'][i:i + batch_size]
  ids_batch = [str(n) for n in range(i, i_end)]
  # create embeddings
  res = openai.Embedding.create(input=lines_batch, engine=MODEL)
  embeds = [record['embedding'] for record in res['data']]
  # prep metadata and upsert batch
  meta = [{'text': line} for line in lines_batch]
  to_upsert = zip(ids_batch, embeds, meta)
  # upsert to Pinecone
  index.upsert(vectors=list(to_upsert))

# ----------

query = "What caused the 1929 Great Depression?"
xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
print(xq)

res = index.query([xq], top_k=5, include_metadata=True)
print(res)

for match in res['matches']:
  print(f"{match['score']:.2f}: {match['metadata']['text']}")

# ----------

query = "What was the cause of the major recession in the early 20th century?"

# create the query embedding
xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

# query, returning the top 5 most similar results
res = index.query([xq], top_k=5, include_metadata=True)

for match in res['matches']:
  print(f"{match['score']:.2f}: {match['metadata']['text']}")

# ----------

query = "Why was there a long-term economic downturn in the early 20th century?"

# create the query embedding
xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

# query, returning the top 5 most similar results
res = index.query([xq], top_k=5, include_metadata=True)

for match in res['matches']:
  print(f"{match['score']:.2f}: {match['metadata']['text']}")

pinecone.delete_index(index_name)

# ---------------------------------------------------------------------------------------------------------
#%%time
# https://github.com/pinecone-io/examples/blob/master/learn/recommendation/article-recommender/article_recommendations.ipynb
# ---------------------------------------------------------------------------------------------------------

import pandas as pd
import re
import openai
import os
import pinecone
import os
import pandas as pd
import numpy as np
import time
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from statistics import mean

# !wget https://www.dropbox.com/s/cn2utnr5ipathhh/all-the-news-2-1.zip -q --show-progress
# !unzip -q all-the-news-2-1.zip

import torch
from sentence_transformers import SentenceTransformer

# Load Pinecone API key
api_key = os.getenv('PINECONE_API_KEY') or 'YOUR_API_KEY'
pinecone.init(
    api_key=api_key,
    environment="YOUR_ENV"  # find next to API key in console
)

index_name = 'articles-recommendation'

# If index of the same name exists, then delete it
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

pinecone.create_index(index_name, dimension=300)

index = pinecone.Index(index_name)
index.describe_index_stats()

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('average_word_embeddings_komninos', device=device)

import torch
from sentence_transformers import SentenceTransformer

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('average_word_embeddings_komninos', device=device)

NROWS = 200000      # number of rows to be loaded from the csv, set to None for loading all rows, reduce if you have a low amount of RAM or want a faster execution
BATCH_SIZE = 500    # batch size for upserting

def prepare_data(data) -> pd.DataFrame:
    'Preprocesses data and prepares it for upsert.'
    
    # add an id column
    print("Preparing data...")
    data["id"] = range(len(data))

    # extract only first few sentences of each article for quicker vector calculations
    data['article'] = data['article'].fillna('')
    data['article'] = data.article.apply(lambda x: ' '.join(re.split(r'(?<=[.:;])\s', x)[:4]))
    data['title_article'] = data['title'] + data['article']
    
    # create a vector embedding based on title and article columns
    print('Encoding articles...')
    encoded_articles = model.encode(data['title_article'])
    data['article_vector'] = pd.Series(encoded_articles.tolist())
    
    return data


def upload_items(data):
    'Uploads data in batches.'
    print("Uploading items...")
    
    # create a list of items for upload
    items_to_upload = [(str(row.id), row.article_vector) for i,row in data.iterrows()]
    
    # upsert
    for i in range(0, len(items_to_upload), BATCH_SIZE):
        index.upsert(vectors=items_to_upload[i:i+BATCH_SIZE])

    
def process_file(filename: str) -> pd.DataFrame:
    'Reads csv files in chunks, prepares and uploads data.'
    
    data = pd.read_csv(filename, nrows=NROWS)
    data = prepare_data(data)
    upload_items(data)
    return data
            
uploaded_data = process_file(filename='all-the-news-2-1.csv')

# -------

# Print index statistics
index.describe_index_stats()

titles_mapped = dict(zip(uploaded_data.id, uploaded_data.title))
sections_mapped = dict(zip(uploaded_data.id, uploaded_data.section))
publications_mapped = dict(zip(uploaded_data.id, uploaded_data.publication))

def get_wordcloud_for_user(recommendations):

    stopwords = set(STOPWORDS).union([np.nan, 'NaN', 'S'])

    wordcloud = WordCloud(
                   max_words=50000, 
                   min_font_size =12, 
                   max_font_size=50, 
                   relative_scaling = 0.9, 
                   stopwords=set(STOPWORDS),
                   normalize_plurals= True
    )

    clean_titles = [word for word in recommendations.title.values if word not in stopwords]
    title_wordcloud = wordcloud.generate(' '.join(clean_titles))

    plt.imshow(title_wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# ------

# first create a user who likes to read sport news about tennis
sport_user = uploaded_data.loc[((uploaded_data['section'] == 'Sports News' ) | 
                                (uploaded_data['section'] == 'Sports')) &
                                (uploaded_data['article'].str.contains('Tennis'))][:10]

print('\nHere is the example of previously read articles by this user:\n')
print(sport_user[['title', 'article', 'section', 'publication']])

# then create a vector for this user
a = sport_user['article_vector']
sport_user_vector = [*map(mean, zip(*a))]

# query the pinecone
res = index.query(sport_user_vector, top_k=10)

# print results
ids = [match.id for match in res.matches]
scores = [match.score for match in res.matches]
df = pd.DataFrame({'id': ids, 
                   'score': scores,
                   'title': [titles_mapped[int(_id)] for _id in ids],
                   'section': [sections_mapped[int(_id)] for _id in ids],
                   'publication': [publications_mapped[int(_id)] for _id in ids]
                    })

print("\nThis table contains recommended articles for the user:\n")
print(df)
print("\nA word-cloud representing the results:\n")
get_wordcloud_for_user(df)

# ------

# first create a user who likes to read news about Xbox
entertainment_user = uploaded_data.loc[((uploaded_data['section'] == 'Entertainment') |
                                        (uploaded_data['section'] == 'Games') |
                                        (uploaded_data['section'] == 'Tech by VICE')) &
                                        (uploaded_data['article'].str.contains('Xbox'))][:10]

print('\nHere is the example of previously read articles by this user:\n')
print(entertainment_user[['title', 'article', 'section', 'publication']])

# then create a vector for this user
a = entertainment_user['article_vector']
entertainment_user_vector = [*map(mean, zip(*a))]

# query the pinecone
res = index.query(entertainment_user_vector, top_k=10)

# print results
ids = [match.id for match in res.matches]
scores = [match.score for match in res.matches]
df = pd.DataFrame({'id': ids, 
                   'score': scores,
                   'title': [titles_mapped[int(_id)] for _id in ids],
                   'section': [sections_mapped[int(_id)] for _id in ids],
                   'publication': [publications_mapped[int(_id)] for _id in ids]
                    })

print("\nThis table contains recommended articles for the user:\n")
print(df)
print("\nA word-cloud representing the results:\n")
get_wordcloud_for_user(df)

# -------

# first create a user who likes to read about Wall Street business news
business_user = uploaded_data.loc[((uploaded_data['section'] == 'Business News')|
                                   (uploaded_data['section'] == 'business')) &
                                   (uploaded_data['article'].str.contains('Wall Street'))][:10]

print('\nHere is the example of previously read articles by this user:\n')
print(business_user[['title', 'article', 'section', 'publication']])

# then create a vector for this user
a = business_user['article_vector']
business_user_vector = [*map(mean, zip(*a))]

# query the pinecone
res = index.query(business_user_vector, top_k=10)

# print results
ids = [match.id for match in res.matches]
scores = [match.score for match in res.matches]
df = pd.DataFrame({'id': ids, 
                   'score': scores,
                   'title': [titles_mapped[int(_id)] for _id in ids],
                   'section': [sections_mapped[int(_id)] for _id in ids],
                   'publication': [publications_mapped[int(_id)] for _id in ids]
                    })

print("\nThis table contains recommended articles for the user:\n")
print(df)
print("\nA word-cloud representing the results:\n")
get_wordcloud_for_user(df)

# -------
pinecone.delete_index(index_name)