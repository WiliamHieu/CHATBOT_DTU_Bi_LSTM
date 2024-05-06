import tensorflow as tf
import json
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from keras.preprocessing.sequence import pad_sequences
import re
import pickle
import random

MAX_SEQUENCE_LENGTH = 43

# Khai báo từ điển để chuyển tiếng việt CÓ dấu về tiếng việt KHÔNG dấu 
patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}
# Khai báo các ký tự đặc biệt để loại bỏ
special_character = r'[!“”"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]'

# Khai báo danh sách các từ cần loại bỏ
stop_words = ['bạn', 'ban', 'anh', 'chị', 'chi', 'em', 'shop', 'bot', 'ad', 'là', 'có', 'và', 'hoặc', 'nếu', 'vậy', 'thế', 'còn']

# Đọc dữ liệu từ tệp chatbot_dataset_CMU.json:
with open(f'data/chatbot_dataset_CMU.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Đọc danh sách lớp từ tệp classes.pkl:

classes = pickle.load(open(f"model/classes.pkl", "rb"))

# Tải mô hình từ tệp model.h5:
model = tf.keras.models.load_model(f'model/model.h5')

# Đọc tokenizer từ tệp tokenizer.pkl:
with open(f"model/tokenizer.pkl", 'rb') as handle:
    tokenizer = pickle.load(handle)


# Hàm làm sạch các ký tự, chuyển thành tiếng việt không dấu và chuyển chữ thành chữ thường
def convert_to_no_accents(text):
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        output = re.sub(regex.upper(), replace.upper(), output)
        output = re.sub(special_character, '', output)
        output = output.lower()
    return output

#  Gắn nhãn dự đoán
def response(tag):
    for i in intents['intents']:
        if i['tag'] == tag:
            print(i)
            return random.choice(i['responses'])

#  Chọn câu trả lời
def classify(sentence):
    message = convert_to_no_accents(sentence)
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=40, dtype='int32', value=0)
    pred = model.predict(padded)
#  Chọn câu trả lời

    tag = classes[np.argmax(pred)]
    result = response(tag)
    return tag, result

# kiểm tra
if __name__=='__main__':
    sentence = 'lịch thi'
    result = classify(sentence)
    print(result)