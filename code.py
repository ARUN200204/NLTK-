!pip install sentencepiece
import requests
from bs4 import BeautifulSoup
import urllib
url1 =urllib.request.urlopen('https://en.wikipedia.org/wiki/History_of_South_India').read()
soup =BeautifulSoup(url1, 'lxml')
# print(soup.prettify)
import requests
from bs4 import BeautifulSoup
import urllib
url1 =urllib.request.urlopen('https://en.wikipedia.org/wiki/Chola_dynasty').read()
soup1 =BeautifulSoup(url1, 'lxml')
# print(soup1.prettify)
for a in soup.find_all("a"):
    a.replace_with(a.text)

text=""
for paragraph  in soup.find_all("p"):
  text +=paragraph.text
print(text)
import re
pattern = r"\[\w+\]"
text = re.sub(pattern, "", text)
text=text.lower()
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

words = text.split()
filtered_words = [word for word in words if word.lower() not in stop_words]
filtered_text = ' '.join(filtered_words)

print(filtered_text)
sentences = nltk.tokenize.sent_tokenize(filtered_text)
print(len(sentences))
!pip install transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
from transformers import BartTokenizer, BartForConditionalGeneration
model_1 = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
tokenizer_1 = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
!nvidia-smi
from transformers import pipeline
summarizer_1= pipeline("summarization",model=model_1,tokenizer=tokenizer_1,device=0)
from transformers import pipeline
summarizer_1= pipeline("summarization",model=model_1,tokenizer=tokenizer_1,device=0)
# initialize
length = 0
chunk = ""
chunks = []
count = -1
tokenizer=tokenizer_1
for sentence in sentences:
  count += 1
  combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

  if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
    chunk += sentence + " " # add the sentence to the chunk
    length = combined_length # update the length counter

    # if it is the last sentence
    if count == len(sentences) - 1:
      chunks.append(chunk.strip()) # save the chunk
    
  else: 
    chunks.append(chunk.strip()) # save the chunk
    
    # reset 
    length = 0 
    chunk = ""

# taking  care of the overflow sentence
    chunk += sentence + " "
    length = len(tokenizer.tokenize(sentence))
len(chunks)
summary=summarizer_1(chunks)
total_summary1=" "
import re
pattern = r"\xa0"
# summary = re.sub(pattern, " ",summary[i] for i in range(len(summary)))
for i in range(len(summary)):
  total_summary=total_summary + summary1[i]['summary_text'].replace("\xa0"," ")
  
