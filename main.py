import pandas as pd
import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# 隐藏Streamlit警告信息
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

# 加载 BERT 模型和分词器
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, bert_model = load_model()

# 读取 CSV 文件
books_file_path = 'BooksTest.csv'
ratings_file_path = 'RatingsTest.csv'

@st.cache_resource
def load_data():
    books_df = pd.read_csv(books_file_path)
    ratings_df = pd.read_csv(ratings_file_path)
    return books_df, ratings_df

books_df, ratings_df = load_data()

# 筛选出 Ratings.csv 中存在于 BooksTest.csv 的 ISBN 记录
filtered_ratings_df = ratings_df[ratings_df['ISBN'].isin(books_df['ISBN'])]

# 生成图书标题嵌入
@st.cache_resource
def generate_embeddings(texts):
    # 如果输入是单个文本，则转换为列表，以便适应tokenizer的要求
    if isinstance(texts, str):
        texts = [texts]

    # 使用tokenizer对文本进行分词和编码
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # 使用BertModel生成文本的嵌入
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # 取得最后一层的隐藏状态，并计算平均值作为嵌入表示
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings


@st.cache_resource
def get_book_embeddings():
    return generate_embeddings(books_df['Book-Title'].values)

books_df['title_embedding'] = get_book_embeddings()

# Streamlit 应用程序
st.title("书籍推荐系统")

input_keyword = st.text_input("请输入关键词以推荐书籍:")
ratings = {}

# 基于评分生成进一步的推荐
def get_further_recommendations(ratings, books_df, filtered_ratings_df):
    # 准备 Surprise 数据集
    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(filtered_ratings_df[['User-ID', 'ISBN', 'Book-Rating']], reader)

    # 训练 SVD 模型
    trainset, testset = train_test_split(data, test_size=0.25)
    svd = SVD()
    svd.fit(trainset)

    # 预测用户评分
    user_id = 999999  # 为了进行预测，假设一个新的用户ID
    for isbn, rating in ratings.items():
        svd.trainset.ur[user_id].append((svd.trainset.to_inner_iid(isbn), rating))

    # 预测评分
    all_books = books_df['ISBN'].unique()
    predicted_ratings = []
    for isbn in all_books:
        if isbn not in ratings.keys():
            predicted_rating = svd.predict(user_id, isbn).est
            predicted_ratings.append((isbn, predicted_rating))

    # 按预测评分排序并推荐前10本书籍
    recommended_isbns = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:10]
    recommended_books = books_df[books_df['ISBN'].isin([isbn for isbn, _ in recommended_isbns])]

    return recommended_books

if st.button("关键词图书推荐") or input_keyword:
    input_embedding = generate_embeddings([input_keyword]).squeeze()

    # 计算相似度
    title_embeddings_matrix = torch.stack(books_df['title_embedding'].tolist()).numpy()
    similarities = cosine_similarity(input_embedding.unsqueeze(0).numpy(), title_embeddings_matrix)
    books_df['similarity'] = similarities.flatten()

    # 找到相似度最高的三本书
    recommended_books = books_df.nlargest(3, 'similarity')

    st.write("推荐书籍:")
    for _, row in recommended_books.iterrows():
        st.write(f"书名: {row['Book-Title']}, 作者: {row['Book-Author']}, 出版年份: {row['Year-Of-Publication']}, 出版社: {row['Publisher']}")
        rating = st.selectbox(f"对 {row['Book-Title']} 评分:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key=f"rating_{row['ISBN']}")
        ratings[row["ISBN"]] = rating

    if st.button("提交评分"):
        st.title("根据您的评分，推荐以下书籍:")
        further_recommended_books = get_further_recommendations(ratings, books_df, filtered_ratings_df)
        if further_recommended_books is not None:
            for _, row in further_recommended_books.iterrows():
                st.write(f"书名: {row['Book-Title']}, 作者: {row['Book-Author']}, 出版年份: {row['Year-Of-Publication']}, 出版社: {row['Publisher']}")
        else:
            st.write("没有足够的数据来推荐更多书籍。")

# 运行应用：在终端中运行 `streamlit run app.py`
