import time
import faiss
import torch
import numpy as np
from pilot_ann import utils, layers
from tqdm import tqdm


def timing_simple():
    # do not use timeit to avoid implicit cacheing
    def decorator_func(func):
        def wrap_func(*args, **kwargs):
            before = time.perf_counter()
            output = func(*args, **kwargs)
            after = time.perf_counter()
            return output, after - before
        return wrap_func
    return decorator_func


class ANNProfiler:
    N_WARMUP = 2
    CHUNK_SIZE = 128
    EF_CONSTRUCT = 80

    def __init__(self,
                 dataset: str,
                 algorithm: str,
                 n_queries: int,
                 d_principle: int,
                 entry_method: str,
                 sample_ratio: float = 0.25):
        self.algorithm = algorithm
        self.n_queries = n_queries
        self.d_principle = d_principle
        self.entry_method = entry_method
        self.sample_ratio = sample_ratio
        assert algorithm in ['hnsw', 'nsg']
        # storage
        self.loader = utils.DataLoader(dataset)
        print('[{}] dataset: {}, d_model: {}'.format(
            self.algorithm, dataset, self.loader.d_model)
        )
        # index
        self.index = None

    @timing_simple()
    def search_faiss(self,
                     index: faiss.Index,
                     query: np.ndarray,
                     k: int, ef_search: int):
        if self.algorithm == 'nsg':
            index.nsg.search_L = ef_search
        elif self.algorithm == 'hnsw':
            index.hnsw.efSearch = ef_search
        else:
            raise NotImplementedError
        return index.search(query, k)[-1]

    @timing_simple()
    def search_hnswlib(self,
                       index: faiss.Index,
                       query: np.ndarray,
                       k: int, ef_search: int):
        if self.algorithm != 'hnsw':
            raise NotImplementedError
        index.set_ef(ef_search)
        return index.knn_query(query, k=k)[0]

    @timing_simple()
    def search_pilot(self,
                     index: layers.IndexStaged,
                     query: torch.Tensor,
                     k: int, ef_search: int):
        return index.search(
            query, k=k, ef_search=ef_search,
            chunk_size=self.CHUNK_SIZE
        )[0].cpu()

    def build(self,
              method: str,
              n_neighbors: int,):
        # init
        storage = self.loader.load_storage()

        # index
        if method == 'faiss':
            if self.algorithm == 'nsg':
                index = faiss.IndexNSGFlat(
                    self.loader.d_model,
                    n_neighbors,
                    faiss.METRIC_L2
                )
                index.GK = self.EF_CONSTRUCT
            elif self.algorithm == 'hnsw':
                index = faiss.IndexHNSWFlat(
                    self.loader.d_model,
                    n_neighbors // 2,
                    faiss.METRIC_L2
                )
                index.hnsw.efConstruction = self.EF_CONSTRUCT
            else:
                raise NotImplementedError
            index.add(storage.numpy())
        elif method == 'hnswlib':
            import hnswlib
            index = hnswlib.Index(
                space='l2', dim=self.loader.d_model
            )
            index.init_index(
                max_elements=self.loader.n_storage,
                ef_construction=self.EF_CONSTRUCT,
                M=n_neighbors // 2
            )
            index.add_items(storage.numpy())
        elif method == 'search-pilot':
            if self.algorithm == 'nsg':
                graph_method = 'nsg{}'.format(n_neighbors)
            elif self.algorithm == 'hnsw':
                graph_method = 'nsw{}'.format(n_neighbors)
            else:
                raise NotImplementedError
            index = layers.IndexStaged(
                d_model=self.loader.d_model,
                d_principle=self.d_principle,
                sample_ratio=self.sample_ratio,
                entry_method=self.entry_method,
                graph_method=graph_method
            )
            index.train(storage)
            index.to(device='cuda')
        else:
            raise NotImplementedError
        self.index = index

    def run(self,
            k: int,
            ef_search: int,
            n_repeats: int = 20):
        # profile
        timings, recalls = [], []
        for _ in tqdm(
            range(n_repeats + self.N_WARMUP), desc='evaluate'
        ):
            # query
            query, target = self.loader.load_query(
                n_queries=self.n_queries, k=k
            )
            time.sleep(2.0)

            # search
            if isinstance(self.index, faiss.Index):
                indices, duration = self.search_faiss(
                    self.index, query=query.numpy(), k=k, ef_search=ef_search
                )
                indices = torch.from_numpy(indices)
            elif isinstance(self.index, layers.IndexStaged):
                indices, duration = self.search_pilot(
                    self.index, query=query, k=k, ef_search=ef_search
                )
            else:
                import hnswlib
                if isinstance(self.index, hnswlib.Index):
                    indices, duration = self.search_hnswlib(
                        self.index, query=query, k=k, ef_search=ef_search
                    )
                    indices = torch.from_numpy(indices.astype('int32'))
                else:
                    raise NotImplementedError

            # timing
            recall = utils.recall(indices, target=target)
            timings.append(1000.0 * duration), recalls.append(recall)

        # show result
        recalls = recalls[self.N_WARMUP:]
        timings = timings[self.N_WARMUP:]
        avg_recall = sum(recalls) / len(recalls)
        avg_timing = sum(timings) / len(timings)
        return avg_recall, avg_timing
