# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %%
df = pd.read_csv("new_orleans_airbnb_listings.csv")
df.head()
# %%
df["host_response_rate"] = df["host_response_rate"].str.replace("%", "").astype(float)
df["host_acceptance_rate"] = df["host_acceptance_rate"].str.replace("%", "").astype(float)
df["price"] = df["price"].str.replace("$", "").str.replace(",", "").astype(float)
# %%
sns.histplot(x=df["price"], bins=200)
plt.xlim(0, 2000)
plt.savefig("price_histogram.png")
plt.show()
# %%
sns.histplot(x=df["availability_30"], bins=30)
plt.savefig("availability_30_histogram.png")
plt.show()
# %%
plt.figure(figsize=(20, 20))
for i, col in enumerate(df.select_dtypes(include="number")):
    plt.subplot(4, 7, i+1)
    sns.violinplot(x=df[col])
    plt.xlim(df[col].quantile(0.01), df[col].quantile(0.99))
plt.savefig("violinplot.png")
plt.show()
# %%
plt.figure(figsize=(15, 15))
plt.scatter(df["longitude"], df["latitude"],
            c=df["review_scores_location"])
plt.colorbar(orientation="horizontal", pad=0.01)
img = plt.imread("new_orleans_map.png")
plt.imshow(img, extent=[-90.15, -89.75, 29.90, 30.15], alpha=1)
plt.savefig("map_scatterplot.png")
plt.show()
# %%
df.select_dtypes(include="object")
# %%
count = 0
plt.figure(figsize=(15, 3))
for i, col in enumerate(df.select_dtypes(include="object")):
    if df[col].nunique() < 3:
        count += 1
        plt.subplot(1, 5, count)
        # make some space between the plots
        plt.subplots_adjust(wspace=0.5)
        sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts())
        plt.xticks(rotation=15)
    
plt.savefig("barplot_0.png")
plt.show()

# %%
count = 0
plt.figure(figsize=(12, 3))
for i, col in enumerate(df.select_dtypes(include="object")):
    if df[col].nunique() > 3 and df[col].nunique() < 10:
        count += 1
        plt.subplot(1, 2, count)
        plt.subplots_adjust(wspace=0.3)
        sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts())
        plt.xticks(rotation=15)
plt.subplots_adjust(hspace=1)
plt.savefig("barplot_1.png")
plt.show()
# %%
plt.figure(figsize=(15, 5))
for col in df.select_dtypes(include="object"):
    if df[col].nunique() > 100:
        continue
    if df[col].nunique() >= 10:
        plt.figure(figsize=(15, 5))
        plt.xticks(rotation=90)
        sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts())
        plt.savefig(f"barplot_{col}.png")
        plt.show()
# %%
plt.figure(figsize=(15, 5))
sns.barplot(x=df["availability_30"], y=df["price"], estimator=np.median)
plt.ylabel("median price")
plt.savefig("availability_30_median.png")
plt.show()
# %%
plt.figure(figsize=(15, 5))
sns.scatterplot(x=df["availability_30"], y=df["price"])
plt.yscale("log")
plt.show()
# %%
sns.scatterplot(x=df["host_response_rate"], y=df["review_scores_communication"])
plt.ylim(4, )
plt.show()
sns.scatterplot(x=df["host_acceptance_rate"], y=df["review_scores_communication"])
plt.ylim(4, 5)
plt.show()

# %%
plt.figure(figsize=(15, 5))
bins = np.arange(0, 110, 10)
x = df.groupby(pd.cut(df["host_acceptance_rate"], bins))["review_scores_communication"].mean().index
y = df.groupby(pd.cut(df["host_acceptance_rate"], bins))["review_scores_communication"].mean()
sns.barplot(x=x, y=y)
plt.ylim(3, 5)
plt.ylabel("mean review_scores_communication")
plt.savefig("host_acceptance_rate_review_scores_communication.png")
plt.show()

# %%
plt.figure(figsize=(15, 5))
bins = np.arange(0, 110, 10)
x = df.groupby(pd.cut(df["host_response_rate"], bins))["review_scores_communication"].mean().index
y = df.groupby(pd.cut(df["host_response_rate"], bins))["review_scores_communication"].mean()
sns.barplot(x=x, y=y)
plt.ylim(3, 5)
plt.ylabel("mean review_scores_communication")
plt.savefig("host_response_rate_review_scores_communication.png")
plt.show()

# %%
plt.figure(figsize=(15, 5))
x = df.groupby("host_response_time")["review_scores_communication"].mean().index
y = df.groupby("host_response_time")["review_scores_communication"].mean()
sns.barplot(x=x, y=y)
plt.ylim(4.6, 5)
plt.ylabel("mean review_scores_communication")
plt.savefig("host_response_time_review_scores_communication.png")
plt.show()
# %%
plt.figure(figsize=(15, 5))
x = df.groupby("neighbourhood_cleansed")["reviews_per_month"].count().index
y = df.groupby("neighbourhood_cleansed")["reviews_per_month"].count()
sns.barplot(x=x, y=y, order=y.sort_values(ascending=False).index)
plt.xticks(rotation=90)
plt.show()
# %%
sns.boxplot(x=df["room_type"], y=df["price"])
plt.ylim(0, 600)
plt.savefig("boxplot_room_type_price.png")
plt.show()

# %%
amenities = set() 
for amenity in df["amenities"]:
    amenity = amenity.replace("[", "").replace("]", "").replace("\"", "")
    amenity = amenity.split(",")
    amenity = [x[1:] for x in amenity if x[0] == " "]
    amenities.update(amenity)

amenities = list(amenities)
amenities
# %%
amenities_dict = {}
for amenity in amenities:
    df[amenity] = df["amenities"].str.contains(amenity)
    amenities_dict[amenity] = (df[df[amenity]]["price"].median(), df[amenity].sum())
# %%
amenities_dict = {k: v for k, v in amenities_dict.items() if not pd.isnull(v[0])}
amenities_dict = dict(sorted(amenities_dict.items(), key=lambda item: item[1][0], reverse=True))
amenities_dict_popular = {k: v for k, v in amenities_dict.items() if v[1] > 10}
# %%
plt.figure(figsize=(15, 5))
x = list(amenities_dict_popular.keys())[:20]
y = [amenities_dict_popular[amenity][0] for amenity in x]
sns.barplot(x=x, y=y)
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.xlabel("amenities")
plt.ylabel("median price")
plt.savefig("amenities_median_price_high.png")
plt.show()
# %%
plt.figure(figsize=(15, 5))
x = list(amenities_dict_popular.keys())[-20:]
y = [amenities_dict_popular[amenity][0] for amenity in x]
sns.barplot(x=x, y=y)
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.xlabel("amenities")
plt.ylabel("median price")
plt.savefig("amenities_median_price_low.png")
plt.show()
# %%
amenities_dict_popular = dict(sorted(amenities_dict_popular.items(), key=lambda item: item[1][1], reverse=True))
plt.figure(figsize=(15, 5))
x = list(amenities_dict_popular.keys())[:20]
y = [amenities_dict_popular[amenity][1] for amenity in x]
sns.barplot(x=x, y=y)
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.xlabel("amenities")
plt.ylabel("count")
plt.savefig("amenities_count.png") 
plt.show()
# %%
plt.figure(figsize=(15, 5))
x = df.groupby("neighbourhood_cleansed")["review_scores_location"].median().index
y = df.groupby("neighbourhood_cleansed")["review_scores_location"].median()
sns.barplot(x=x, y=y, order=y.sort_values(ascending=False).index)
plt.ylabel("median review_scores_location")
plt.xticks(rotation=90)
plt.savefig("neighbourhood_cleansed_review_scores_location.png")
plt.show()
# %%
plt.figure(figsize=(15, 5))
order = df["price"].groupby(df["room_type"]).median().sort_values(ascending=False).index
sns.barplot(x="room_type", y="price", data=df, order=order, estimator=np.median)
plt.show()
# %%
plt.figure(figsize=(15, 5))
order = df["price"].groupby(df["property_type"]).mean().sort_values(ascending=False).index
sns.barplot(x="property_type", y="price", data=df, order=order)
plt.xticks(rotation=90)
plt.show()
# %%
df["title_length"] = df["name"].str.len()
df["description_length"] = df["description"].str.len()
# %%
plt.figure(figsize=(5, 3))
sns.violinplot(x=df["title_length"], bins=20)
plt.show()
# %%
plt.figure(figsize=(5, 3))
sns.violinplot(x=df["description_length"], bins=20)
plt.show()
# %%
bins = pd.cut(df["title_length"], bins=10)
x = df.groupby(bins)["price"].median().index
y = df.groupby(bins)["availability_30"].median()
z = df.groupby(bins)["price"].median()
sns.barplot(x=x, y=y, hue=z, color="red")
plt.legend(title="median price", loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.show()


# %%
bins = pd.cut(df["description_length"], bins=10)
x = df.groupby(bins)["price"].median().index
y = df.groupby(bins)["availability_30"].median()
z = df.groupby(bins)["price"].median()
sns.barplot(x=x, y=y, hue=z, color="red")
plt.legend(title="median price", loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.show()
# %%
df["title_word_count"] = df["name"].str.split().str.len()
df["description_word_count"] = df["description"].str.split().str.len()
# %%
plt.figure(figsize=(15, 5))
sns.violinplot(x=df["title_word_count"], bins=20)
plt.show()
# %%
plt.figure(figsize=(15, 5))
sns.violinplot(x=df["description_word_count"], bins=20)
plt.show()
# %%
bins = pd.cut(df["title_word_count"], bins=10)
x = df.groupby(bins)["price"].median().index
y = df.groupby(bins)["price"].median()
sns.barplot(x=x, y=y)
plt.xticks(rotation=90)
plt.show()

# %%
bins = pd.cut(df["description_word_count"], bins=10)
x = df.groupby(bins)["price"].median().index
y = df.groupby(bins)["price"].median()
sns.barplot(x=x, y=y)
plt.xticks(rotation=90)
plt.show()

# %%
sns.barplot(x="host_is_superhost", y="availability_30", data=df, ax=plt.subplot(2, 3, 1), estimator=np.median)
sns.barplot(x="host_has_profile_pic", y="availability_30", data=df, ax=plt.subplot(2, 3, 2), estimator=np.median)
plt.ylabel("")
plt.yticks([])
sns.barplot(x="host_identity_verified", y="availability_30", data=df, ax=plt.subplot(2, 3, 3), estimator=np.median)
plt.ylabel("")
plt.yticks([])

plt.show()
# %%
sns.scatterplot(x="bedrooms", y="beds", hue="price", data=df, hue_norm=(0, 2000))
plt.savefig("bedrooms_beds_price.png")
plt.show()
# %%
sns.heatmap(df[["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"]].corr(), annot=True)
plt.show()