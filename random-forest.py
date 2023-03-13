#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, ensemble, metrics
from sklearn import inspection
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
#%%
X = pd.read_csv("data/x_train.csv")
y = pd.read_csv("data/y_train.csv")

X = X.sort_index(axis=1)

X_train, X_val, y_train, y_val = model_selection.train_test_split(
    X, y, test_size=0.25, random_state=1)


def train_random_forest(X_train, y_train):
    model = ensemble.RandomForestClassifier(verbose=False,
                                            n_jobs=4,
                                            max_depth=29)
    model.fit(X_train, y_train)
    return model


#%%
full_rf = train_random_forest(X_train, y_train["class"].ravel())
y_hat = full_rf.predict(X_val)
accuracy = metrics.accuracy_score(y_val["class"].ravel(), y_hat)
print(f"Vanilla Random Forest has accuracy {accuracy}")

#%%


def plot_permutation_importances(clf, X_val, y_val, n_top):
    perm_result = inspection.permutation_importance(
        clf, X_val, y_val, n_jobs=4, n_repeats=5)
    mean_importances = pd.Series(perm_result.importances_mean,
                                 index=X_val.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    plt.yscale("log")
    plt.xticks(rotation='vertical')
    plt.bar(height=mean_importances[0:n_top], x=mean_importances[0:n_top].index)
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    return fig, ax


#%%
fig, ax = plot_permutation_importances(full_rf, X_val, y_val["class"].ravel(), 10)
ax.set_title("Permutation importance on all features")
fig.tight_layout()
fig.savefig("permutation-importances.png")
fig.show()
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
plt.savefig("correlation.png", dpi=600)
#%%
# t is the maximum inter-cluster distance allowed.
# small t = many clusters, large t = few clusters
cluster_ids = hierarchy.fcluster(dist_linkage, 0.05, criterion="distance")

# this creates a map (cluster_id) -> (included_features)
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)

# finally, for each cluster, we select one feature that
# will be used for modeling.
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
X_train_sel = X_train.iloc[:, selected_features]
X_val_sel = X_val.iloc[:, selected_features]

#%% DO THE PERMUTATION TEST ON THE SUBSET OF FEATURES
clf_select = train_random_forest(X_train_sel, y_train["class"].ravel())
y_hat_select = clf_select.predict(X_val_sel)
print(
    "Accuracy on test data with features removed: {:.5f}".format(
        metrics.accuracy_score(y_hat_select, y_val)
    )
)
#%%
fig, ax = plot_permutation_importances(clf_select, X_val_sel, y_val, n_top=10)
ax.set_title("Permutation result on selected features")
fig.tight_layout()
plt.savefig("permutation-importances-selected.png")
plt.show()
#%%
gini_importances = pd.Series(full_rf.feature_importances_, index=X_train.columns)
gini_importances = gini_importances.sort_values(ascending=False)
fig, ax = plt.subplots()
plt.xticks(rotation='vertical')
plt.bar(height=gini_importances[0:10], x=gini_importances[0:10].index)
fig.tight_layout()
plt.savefig("gini-importances.png")
plt.show()
#%% Now we'll do something else:
N_ROUNDS = 5
full_features = list(X_train.columns)
clean_features = {x.split("_")[0] for x in full_features}
original_score = full_rf.score(X_val, y_val)
result = pd.DataFrame(index=list(clean_features), columns=range(0,N_ROUNDS))

for i in range(0, N_ROUNDS):
    for f in clean_features:
        relevant_features = [x for x in full_features if (f + "_") in x]
        if len(relevant_features) == 0:
            relevant_features.append(f)
        permutation = np.random.permutation(range(0,X_val.shape[0]))
        perm_X_val = X_val.reset_index(drop=True)
        perm_X_val[relevant_features] = perm_X_val[relevant_features].sample(frac=1, ignore_index=True)
        result.loc[f, i] = full_rf.score(perm_X_val, y_val)

result = original_score - result
#%%

mean_differences = result.mean(axis=1)
fig, ax = plt.subplots()
plt.yscale("log")
plt.xticks(rotation = "vertical")
ax.bar(height = mean_differences, x= mean_differences.index)
plt.tight_layout()
plt.savefig("permutation-importance-handmade.png")
#%%
