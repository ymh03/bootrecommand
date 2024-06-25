import pandas as pd

# 读取 CSV 文件
books_file_path = 'BooksTest.csv'
ratings_file_path = 'Ratings.csv'

books_df = pd.read_csv(books_file_path)
ratings_df = pd.read_csv(ratings_file_path)

# 筛选出 `Ratings.csv` 中存在于 `BooksTest.csv` 的 ISBN 记录
filtered_ratings_df = ratings_df[ratings_df['ISBN'].isin(books_df['ISBN'])]

# 另存为 `RatingsTest.csv`
filtered_ratings_file_path = 'RatingsTest.csv'
filtered_ratings_df.to_csv(filtered_ratings_file_path, index=False)

print(f"筛选后的记录已保存为 {filtered_ratings_file_path}")