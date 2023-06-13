import pandas as pd
import spacy
import random

nlp = spacy.load('en_core_web_sm')

# data = pd.read_csv('/data0/datasets/MSCOCO/test/test.csv')
data = pd.read_csv('/data0/datasets/FLICKR30K/test/flickr30k.csv')

shuffle_naa = []
shuffle_tri = []
shuffle_within_tri = []
captions = []

images = data['image']
original_captions = data['caption']

for caption in original_captions:
    caption = caption.replace('.','')
    caption = caption.lower()
    captions.append(caption)
    doc = nlp(caption)

    adj, noun = [], []

    for token in doc:
        if token.pos_ == 'ADJ':
            adj.append(token.text)
        elif token.pos_ == 'NOUN':
            noun.append(token.text)

    random.shuffle(adj)
    random.shuffle(noun)
    shuffle_noun_and_adjectives = ''
    for token in doc:
        if token.pos_ == 'ADJ':
            shuffle_noun_and_adjectives += ' '+adj.pop()
        elif token.pos_ == 'NOUN':
            shuffle_noun_and_adjectives += ' '+noun.pop()
        else:
            shuffle_noun_and_adjectives += ' '+token.text

    if shuffle_noun_and_adjectives[1:] != caption:
        shuffle_naa.append(shuffle_noun_and_adjectives[1:])
    else:
        shuffle_naa.append(float('nan'))

    shuffle_trigrams = []
    for i in range(0, len(doc), 3):
        shuffle_trigrams.append(doc[i:i+3].text)
    random.shuffle(shuffle_trigrams)

    neg_caption = ' '.join(shuffle_trigrams)
    if neg_caption != caption:
        shuffle_tri.append(neg_caption)
    else:
        shuffle_tri.append(float('nan'))

    shuffle_words_within_each_trigram = []
    for i in range(0, len(doc), 3):
        trigrams = list((doc[i:i+3].text).split())
        random.shuffle(trigrams)
        shuffle_words_within_each_trigram.append(' '.join(trigrams))

    neg_caption = ' '.join(shuffle_words_within_each_trigram)
    if neg_caption != caption:
        shuffle_within_tri.append(neg_caption)
    else:
        shuffle_within_tri.append(float('nan'))

## Convert to df and save as csv
df = pd.DataFrame({
    'image': images,
    'original_caption': captions,
    'shuffle_noun_and_adjectives': shuffle_naa,
    'shuffle_trigrams': shuffle_tri, 
    'shuffle_words_within_trigrams': shuffle_within_tri
    })

# df.to_csv('mscoco_order_neg.csv', index=False, sep='\t')
df.to_csv('flickr_order_neg.csv', index=False, sep='\t')
