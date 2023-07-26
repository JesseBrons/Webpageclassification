import re
import pandas as pd

with open('content.rdf.u8', 'r') as fl_in:
    lines = [str(line) for line in fl_in.readlines()[:]]

titles = list(filter(None, [re.findall('<d:Title>(.+)</d:Title>', line) for line in lines]))
titles = [x for y in titles for x in y]

urls = list(filter(None, [re.findall('<ExternalPage about="(.+)">', line) for line in lines]))
urls = [x for y in urls for x in y]

topics = list(filter(None, [re.findall('<topic>(.+)</topic>', line) for line in lines]))
topics = [x for y in topics for x in y]
topics_level1 = [topic.split("/")[1] for topic in topics]

df = pd.DataFrame(columns=['category', 'title', 'url'])

df.category = topics_level1
df.title = titles
df.url = urls

df = df.drop(df[df.category == 'World'].index)
df = df.drop(df[df.category == 'Regional'].index)

df.to_csv("./data_dmoz.csv")