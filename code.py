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

for paragraph  in soup1.find_all("p"):
  text1 +=paragraph.text
#second wikipedia site

import re
pattern1 = r"\[\w+\]"
text1 = re.sub(pattern1, "", text1)
text1=text1.lower()
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
words = word_tokenize(text1)

filtered_words = []

for word in words:
    if word.lower() not in stopwords.words('english'):
        filtered_words.append(word)

filtered_text1 = ' '.join(filtered_words)

length = len(filtered_text1)
print(length)
sentences1 = nltk.tokenize.sent_tokenize(filtered_text1)
print(len(sentences1))
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
model_2 = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
tokenizer_2 = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
from transformers import pipeline
summarizer_2= pipeline("summarization",model=model_2,tokenizer=tokenizer_2,device=0)
# initialize
length = 0
chunk = ""
chunks = []
count = -1
tokenizer=tokenizer_1
for sentence in sentences1:
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
    
summary1=summarizer_2(chunks)
total_summary1=" "
import re
pattern = r"\xa0"
# summary = re.sub(pattern, " ",summary[i] for i in range(len(summary)))
for i in range(len(summary1)):
  total_summary1=total_summary1 + summary1[i]['summary_text'].replace("\xa0"," ")
final_summary= total_summary+total_summary
#deep learning
!nvidia-smi
! pip install -q langchain transformers sentence_transformers llama-index
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext
import torch
from langchain.llms.base import LLM
from transformers import pipeline

class customLLM(LLM):
    model_name = "google/flan-t5-large"
    pipeline = pipeline("text2text-generation", model=model_name, device=0, model_kwargs={"torch_dtype":torch.bfloat16})

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]
 
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"

hfemb = HuggingFaceEmbeddings()
embed_model = LangchainEmbedding(hfemb)
from llama_index import Document

text_list = [final_summary]

documents = [Document(t) for t in text_list]
num_output = 400
max_input_size = 1024
max_chunk_overlap = 20



prompt_helper = PromptHelper(max_input_size, num_output,max_chunk_overlap  )
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model,prompt_helper=prompt_helper)
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
import logging

logging.getLogger().setLevel(logging.CRITICAL)
response1 = index.query( "") 
llm_predictor = LLMPredictor(llm=customLLM())


  
