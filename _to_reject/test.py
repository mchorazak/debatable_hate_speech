# Get the tokens with tokenize function
from gensim.utils import tokenize
line = "The hotel  had a really good service!!"
print(list(tokenize(line, deacc=True)))
print(list(tokenize(line, deacc=True)))