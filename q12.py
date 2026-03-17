import numpy as np

def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    
    # Randomly initialize centroids
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    
    for _ in range(max_iters):
        # Compute distances
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # Assign clusters
        labels = np.argmin(distances, axis=1)
        
        # Compute new centroids
        new_centroids = np.array([
            X[labels == i].mean(axis=0) for i in range(k)
        ])
        
        # Stop if converged
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
        
    return centroids, labels

def kmedoids(X, k, max_iters=100):
    n_samples = X.shape[0]
    
    # Initialize medoids randomly
    medoid_indices = np.random.choice(n_samples, k, replace=False)
    medoids = X[medoid_indices]
    
    for _ in range(max_iters):
        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_medoids = []
        
        for i in range(k):
            cluster_points = X[labels == i]
            
            # Compute total distance for each point in cluster
            costs = []
            for point in cluster_points:
                cost = np.sum(np.linalg.norm(cluster_points - point, axis=1))
                costs.append(cost)
            
            # Select point with minimum total distance
            new_medoids.append(cluster_points[np.argmin(costs)])
        
        new_medoids = np.array(new_medoids)
        
        if np.all(medoids == new_medoids):
            break
        
        medoids = new_medoids
    
    return medoids, labels

def fuzzy_c_means(X, c, m=2, max_iters=100, epsilon=1e-5):
    n_samples = X.shape[0]
    
    # Initialize membership matrix randomly
    U = np.random.dirichlet(np.ones(c), size=n_samples)
    
    for _ in range(max_iters):
        U_old = U.copy()
        
        # Compute cluster centers
        centers = []
        for j in range(c):
            numerator = np.sum((U[:, j] ** m).reshape(-1, 1) * X, axis=0)
            denominator = np.sum(U[:, j] ** m)
            centers.append(numerator / denominator)
        centers = np.array(centers)
        
        # Update membership matrix
        for i in range(n_samples):
            for j in range(c):
                distances = np.linalg.norm(X[i] - centers, axis=1)
                U[i, j] = 1 / np.sum(
                    (distances[j] / distances) ** (2 / (m - 1))
                )
        
        # Check convergence
        if np.linalg.norm(U - U_old) < epsilon:
            break
    
    return centers, U

X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
])

centroids, labels = kmeans(X, k=2)

print("K-Means Centroids:\n", centroids)
print("Cluster Labels:\n", labels)

medoids, labels = kmedoids(X, k=2)

print("K-Medoids:\n", medoids)
print("Cluster Labels:\n", labels)

centers, U = fuzzy_c_means(X, c=2, m=2)

print("Fuzzy C-Means Centers:\n", centers)
print("Membership Matrix:\n", U)
