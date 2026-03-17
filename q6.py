import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

#part B
anime = pd.read_csv('Anime_Dataset/anime.csv')
rating = pd.read_csv('Anime_Dataset/cleaned_rating.csv')

data = pd.merge(rating, anime, on="anime_id")

# train test splitting
train, test = train_test_split(data, test_size=0.2)

user_item_matrix = data.pivot_table(
    index='user_id',
    columns='name',
    values='rating_x'
)

user_item_matrix = user_item_matrix.fillna(0)

#part C
user_similarity = cosine_similarity(user_item_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)


def recommend_user_based(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    similar_users = user_similarity_df[user_id]

    weighted_sum = user_item_matrix.T.dot(similar_users)

    similarity_sum = similar_users.sum()

    predicted_ratings = weighted_sum / similarity_sum

    already_rated = user_item_matrix.loc[user_id]
    predicted_ratings = predicted_ratings[already_rated == 0]

    recommended = predicted_ratings.sort_values(ascending=False)

    return recommended.head(top_n)

#part d
item_similarity = cosine_similarity(user_item_matrix.T)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

def recommend_item_based(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]

    scores = item_similarity_df.dot(user_ratings)

    already_rated = user_ratings[user_ratings > 0]
    scores = scores.drop(already_rated.index)

    return scores.sort_values(ascending=False).head(top_n)

#part e
def hybrid_recommend(user_id, top_n=5):
    user_rec = recommend_user_based(user_id, top_n=20)
    item_rec = recommend_item_based(user_id, top_n=20)

    hybrid_scores = (user_rec.add(item_rec, fill_value=0)) / 2

    return hybrid_scores.sort_values(ascending=False).head(top_n)

#part f
def evaluate_hit_rate(recommend_function, k=5):

    hits = 0
    total = 0

    for user in data["user_id"].unique():

        user_data = data[data["user_id"] == user]

        if len(user_data) < 2:
            continue

        # Hide one item
        hidden = user_data.sample(1, random_state=42)
        hidden_item = hidden["name"].values[0]

        remaining = user_data.drop(hidden.index)

        # Build temporary matrix
        temp_matrix = remaining.pivot_table(
            index="user_id",
            columns="name",
            values="rating_x"
        ).fillna(0)

        if user not in temp_matrix.index:
            continue

        # Use existing similarity (simplified for demo)
        recs = recommend_function(user, top_n=k)

        if isinstance(recs, pd.Series):
            if hidden_item in recs.index:
                hits += 1

        total += 1

    if total == 0:
        return 0

    return hits / total

#recommendations
sample_user = user_item_matrix.index[0]

print("\nUser-Based Recommendations:")
print(recommend_user_based(sample_user))

print("\nItem-Based Recommendations:")
print(recommend_item_based(sample_user))

print("\nHybrid Recommendations:")
print(hybrid_recommend(sample_user))

#evaluate model
print("\nEvaluating Models...")

print("User-Based Hit@5:",
      evaluate_hit_rate(recommend_user_based, 5))

print("Item-Based Hit@5:",
      evaluate_hit_rate(recommend_item_based, 5))

print("Hybrid Hit@5:",
      evaluate_hit_rate(hybrid_recommend, 5))

