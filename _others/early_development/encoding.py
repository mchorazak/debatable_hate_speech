# This file contains a script encoding data using a tensorflow encoder.
# The file was developed early and presents an approach different from final.
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from pylab import rcParams
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import tensorflow_hub as hub
from wordcloud import WordCloud, STOPWORDS

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def show_clouds(df):
  hate_text = df[df.label == "hate"]
  non_hate_text = df[df.label == "nothate"]

  hate_speech = " ".join(hate_text.text.to_numpy().tolist())
  non_hate_speech = " ".join(non_hate_text.text.to_numpy().tolist())

  hate_cloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(hate_speech)
  not_hate_cloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(non_hate_speech)

  def show_word_cloud(cloud, title):
    plt.figure(figsize=(16, 10))
    plt.imshow(cloud, interpolation='bilinear')
    plt.title(title)
    plt.axis("off")
    plt.show()

  show_word_cloud(hate_cloud, "Common words in hate speech.")
  show_word_cloud(not_hate_cloud, "Common words in non-hate speech")

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

palette = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(palette))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.test.is_gpu_available()

df = pd.read_csv("../../data/Dynamically Generated Hate Dataset v0.2.3.csv")
df.shape


sns.countplot(x='label',data=df,order=df.label.value_counts().index)

plt.xlabel("type")
plt.title("label")
plt.show()

show_clouds(df)

print("Loading encoder... ")
use = hub.load("encoder/")
print("Encoder loaded.")

from sklearn.preprocessing import OneHotEncoder

type_one_hot = OneHotEncoder(sparse=False).fit_transform(
  df.label.to_numpy().reshape(-1, 1)
)

encoded_list = []
for r in tqdm(df.text):
  emb = use(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  encoded_list.append(review_emb)

encoded_df = pd.DataFrame(encoded_list)
encoded_df = encoded_df.reset_index(drop=True)
df = df.reset_index(drop=True)
dat1 = pd.concat([df, encoded_df], axis=1)

dat1.to_csv('data_encoded.csv', index=False)
print("Data saved to file.")
