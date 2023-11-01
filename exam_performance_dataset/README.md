Key features of this dataset:

    - Sample variance is 0.17
    - Mean Categorical Data Variance is 0.04
    - Within Cluster Variance is 0.02

Results:

    Preproceeding without embeddings:
    - Davies Bouldin score is 1.49
    - Silhouette Score is 0.19
    
    Preproceeding with embeddings:
    - Davies Bouldin Score is 1.71
    - Silhouette Score: 0.22

### Application of embeddingns on this dataset has *not* created any extra value for clustering with kmenas due to to high sample variance of this data. The results with data encoded with embeddings is even worse then with classical way to proceed data
