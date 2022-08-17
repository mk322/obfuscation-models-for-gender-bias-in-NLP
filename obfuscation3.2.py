import pandas as pd
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 

with open('male.txt', encoding='utf-8') as f1:
    male_words = dict.fromkeys(f1.read().splitlines(), 'VOO')
    f1.close()

with open('female.txt', encoding='utf-8') as f2:
    female_words = dict.fromkeys(f2.read().splitlines(), 'VOO')
    f2.close()

for word in male_words.keys():
    if word in model.key_to_index:
        similarity_dic = {}
        for fem_word in female_words.keys():
            if fem_word in model.key_to_index:
                similarity_dic[fem_word] = model.similarity(word, fem_word)
        if max(similarity_dic.values()) >= 0.5:
            male_words[word] = max(similarity_dic, key=similarity_dic.get)

print('Half Done')

for word in female_words.keys():
    if word in model.key_to_index:
        similarity_dic = {}
        for male_word in male_words.keys():
            if male_word in model.key_to_index:
                similarity_dic[male_word] = model.similarity(word, male_word)
        if max(similarity_dic.values()) >= 0.5:
            female_words[word] = max(similarity_dic, key=similarity_dic.get)


dataset = pd.read_csv('dataset.csv', header=0)
male_text = dataset.loc[dataset['op_gender'] == 'M', 'post_text']
female_text = dataset.loc[dataset['op_gender'] == 'W', 'post_text']
male_text_c = male_text.copy()
female_text_c = female_text.copy()

for i, v in male_text.items():
    new_text = ''
    words = v.split()
    for word in words:
        if word in male_words and male_words[word] != 'VOO':
            new_text += male_words[word] + ' '
        else:
            new_text += word + ' '
    male_text_c.update(pd.Series([new_text[:-1]], index=[i]))

for i, v in female_text.items():
    new_text = ''
    words = v.split()
    for word in words:
        if word in female_words and female_words[word] != 'VOO':
            new_text += female_words[word] + ' '
        else:
            new_text += word + ' '
    female_text_c.update(pd.Series([new_text[:-1]], index=[i]))

dataset_c = dataset.copy()
dataset_c['post_text'].update(male_text_c.append(female_text_c))
dataset_c.to_csv(r'step3.2.csv', index = False)
print('Finished')