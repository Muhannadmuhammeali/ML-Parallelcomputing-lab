import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

# B. Load and Examine Dataset
df = pd.read_csv('Groceries_dataset.csv')
print("Dataset Head and Info")
print(df.head(5))
print(df.info())

transactions = (
    df.groupby(['Member_number', 'Date'])['itemDescription']
    .apply(list)
    .tolist()
)
# C. Generating Itemsets
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print("\nEncoded Transaction Data:")
print(df_encoded.head())

# Calculate support for individual items
item_support = df_encoded.mean()
print("\nSupport of Individual Items:")
print(item_support)

# D. Identifying Frequent Itemsets (Apriori)
min_support = 0.0001  

frequent_itemsets = apriori(
    df_encoded,
    min_support=min_support,
    use_colnames=True
)


print("\nFrequent Itemsets:")
print(frequent_itemsets)

# E. Deriving Association Rules
# Minimum confidence threshold
min_confidence = 0.05

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=min_confidence
)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# F. Evaluation of Association Rules

# Filter strong rules (high confidence and lift > 1)
strong_rules = rules[
    (rules['confidence'] >= min_confidence) &
    (rules['lift'] > 1)
]
print("\nStrong Association Rules:")
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
