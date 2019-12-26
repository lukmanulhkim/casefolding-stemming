# Create by Muhammad Azmi
#2019
#for case folding and stemming
import re
import string

import matplotlib.pyplot as plt
import nltk.probability
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# from nltk.tokenize import sent_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stop_factory = StopWordRemoverFactory().get_stop_words() #load defaul stopword
more_stopword = ['daring', 'online'] #menambahkan stopword

#baca file, ekstensi yang dibaca TXT, JSON dll. Silahkan ganti output10.txt dengan teks hasil crawling atau scraping anda
file_input = open("output9.json", "r",encoding="utf8")

#tahapan case folding
kalimat = file_input.read()

#hilangkan huruf besar
lower_case = kalimat.lower()

#hilangkan url
pattern = r"http\S+"
text = lower_case
removeurl = re.sub(pattern,"",lower_case)

#hilangkan angka
hasil = re.sub(r"\d+", "",removeurl)




# re.sub(r'http\S+',",stringliteral)
#hilangkan tanda baca
hasil1 = hasil.translate(str.maketrans("","",string.punctuation))
#hilangkan spasi
hasil2 = hasil1.strip()

#Tokenizing

hasil2 = hasil2.translate(str.maketrans('', '', string.punctuation)).lower()
#tokenizing kata
tokens = nltk.tokenize.word_tokenize(hasil2)
#tokenizing kalimat
tokens1 = nltk.tokenize.sent_tokenize(hasil2)
kemunculan = nltk.FreqDist(tokens)

print(kemunculan.most_common())

# print(tokens1)
kemunculan.plot(30,cumulative=False)
plt.show()
#filtering with nltk
tokens2 = word_tokenize(hasil2)
# listStopword = set(stopwords.words('indonesian'))

# removed = []
# for t in tokens2:
#     if t not in listStopword:
#         removed.append(t)

# print(removed)
#filtering with sastrawi
stop = stopword.remove(hasil2)
tokens2 = nltk.tokenize.word_tokenize(stop)
print(tokens2)

data = stop_factory + more_stopword  # menggabungkan stopword

dictionary = ArrayDictionary(data)
str = StopWordRemover(dictionary)
tokens2 = nltk.tokenize.word_tokenize(str.remove(hasil2))
#stemming bahasa indo
factory = StemmerFactory()
stemmer = factory.create_stemmer()
hasil4 = stemmer.stem(hasil2)
print(hasil4)
#simpan ke file txt, output berupa file txt yang disimpan pada direktori stemming (silahkan buat direktori stemming
#pada laptop/komputer anda
file_output = open("stemming/hasil15.txt", "w")

# tulis teks ke file
file_output.write(hasil4)

# tutup file
file_output.close()
#print(tokens2, file=open("hasil.txt", "a"))
# lower_case = kalimat.lower()
# print(lower_case)
file_input.close()
