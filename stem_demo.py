import re
from nltk import LancasterStemmer
from nltk.tokenize import word_tokenize

# nltk.download('punkt_tab') # Chỉ cần tải lần đầu

stemmer = LancasterStemmer()
input = "Greek authorities deployed 130 firefighters, 12 planes and 12 helicopters to battle the blaze over 420.5 hecta of land."

# Mã hóa số -> NUM (regex)
str_encrypted_num = re.sub(r'[0-9]+.?[0-9]+', 'NUM', input)
print('Step 1:')
print(str_encrypted_num)

# loại bỏ các dấu chấm hoặc phẩy ở đầu hoặc cuối mỗi từ (regex)
str_remove_comma = re.sub(r'([.,]$|[.,] )|(^[.,]| [.,])', ' ', str_encrypted_num)
print('Step 2:')
print(str_remove_comma)

lst_word_stemmed = []
words = word_tokenize(str_remove_comma) # Tách từ

for word in words:
	stemmed_word = stemmer.stem(word) # Xử lý Stemming cho từng từ
	if stemmed_word == "num": # Vì hàm stemming làm lower case mất chữ 'num' nên phải uppercase lại
		stemmed_word = "NUM"
	lst_word_stemmed.append(stemmed_word)

print("Final Output: ")
print(lst_word_stemmed)
