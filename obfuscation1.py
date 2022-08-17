import pandas as pd
import random

with open('male.txt', encoding='utf-8') as f1:
    male_words = dict.fromkeys(f1.read().splitlines(), 1)
    f1.close()

with open('female.txt', encoding='utf-8') as f2:
    female_words = dict.fromkeys(f2.read().splitlines(), 1)
    f2.close()

dataset = pd.read_csv('dataset.csv', header=0)

male_text = dataset.loc[dataset['op_gender'] == 'M']['post_text']
female_text = dataset.loc[dataset['op_gender'] == 'W']['post_text']

for i, v in male_text.items():
    new_text = ''
    words = v.split()
    for word in words:
        if word in male_words:
            new_text += random.choice(list(female_words.keys())) + ' '
        else:
            new_text += word + ' '
    male_text[i] = new_text[:-1]

for i, v in female_text.items():
    new_text = ''
    words = v.split()
    for word in words:
        if word in female_words:
            new_text += random.choice(list(male_words.keys())) + ' '
        else:
            new_text += word + ' '
    female_text[i] = new_text[:-1]

dataset['post_text'] = male_text.append(female_text)

dataset.to_csv(r'step1.csv', index = False)