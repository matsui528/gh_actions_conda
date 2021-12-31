import faiss
import numpy as np

D = 128
N = 10000
X = np.random.random((N, D)).astype(np.float32)  # inputs of faiss must be float32

# Setup
index = faiss.IndexFlatL2(D)
index.add(X)

# Search
topk = 4
dists, ids = index.search(x=X[:3], k=topk)  # Use the top three vectors for querying
print(type(dists), dists.dtype, dists.shape)  # <class 'numpy.ndarray'> float32 (3, 4)
print(type(ids), ids.dtype, ids.shape)  # <class 'numpy.ndarray'> int64 (3, 4)

# Show params
print("N:", index.ntotal)
print("D:", index.d)