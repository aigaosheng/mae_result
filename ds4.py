from llama_index.embeddings import HuggingFaceEmbedding
import pandas as pd
import numpy as np

#load comments
ds4_file = "DS4-assessment-FOMC-Comments.xlsx"
df_comments = pd.read_excel(ds4_file).to_numpy()
# comments = df_comments["comment"]

#convert sentiment word to sentence
class_name = [
    ('5: Hawkish', 'The sentiment of comment  is Hawkish'),
    ('4: Mostly Hawkish', 'The sentiment of comment is Mostly Hawkish'),
    ('3: Neutral', 'The sentiment of comment is Neutral'),
    ('2: Mostly Dovish', 'The sentiment of comment is Mostly Dovish'),
    ('1: Dovish', 'The sentiment of comment is Dovish'),
]
# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# comment embedding
emb_comment = []

for user, comment in df_comments:
    embeddings = embed_model.get_text_embedding(comment)
    emb_comment.append(embeddings)

# comment embedding
emb_class = []

for _, comment in class_name:
    embeddings = embed_model.get_text_embedding(comment)
    emb_class.append(embeddings)

emb_comment = np.array(emb_comment)
emb_class = np.array(emb_class)
score = np.matmul(emb_comment, emb_class.T)
pred_class = score.argmax(axis = 1)

pred_class_name = list(map(lambda x: class_name[x][0], pred_class))
pred_class_score = []
for k, ik in enumerate(pred_class):
    pred_class_score.append(round(score[k, ik], 4))

user_name = df_comments[:, 0]

with open('./ds4_result.csv', "wt") as o_ft:
    o_ft.write("Member,Sentiment,score\n")
    for m, s, sc in zip(pred_class_name, user_name, pred_class_score):
        o_ft.write(f"{m},{s},{sc}\n")


