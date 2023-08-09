import os
import faiss
import time
import numpy as np
import pickle as pkl
from tqdm import tqdm

def load_embeds(embed_path, dstore_size, dimension, dtype):
    assert os.path.exists(embed_path), embed_path
    return np.memmap(embed_path,
                     dtype=dtype,
                     mode="r",
                     shape=(dstore_size, dimension))

class DataStore(object):
    def __init__(self,
                 embed_path,
                 index_path,
                 trained_index_path,
                 prev_index_path,
                 prev_embed_paths,
                 dstore_size,
                 embeds=None,
                 dimension=2048,
                 dtype=np.float16,
                 ncentroids=4096,
                 code_size=64,
                 probe=8,
                 num_keys_to_add_at_a_time=1000000,
                 DSTORE_SIZE_BATCH=51200000
                 ):

        self.embed_path = embed_path
        self.index_path = index_path
        self.prev_index_path = prev_index_path
        self.trained_index_path = trained_index_path
        self.cuda = True

        self.dstore_size = dstore_size
        self.dimension = dimension
        self.ncentroids = ncentroids
        self.code_size = code_size
        self.probe = probe
        self.num_keys_to_add_at_a_time = num_keys_to_add_at_a_time

        if embeds is not None:
            assert embeds.shape == (dstore_size, dimension)
            self.embs = embeds

        elif embed_path is not None and os.path.exists(embed_path):
            print ("Loading embeds (%d, %d) from %s" % (dstore_size, dimension, embed_path))
            self.embs = load_embeds(embed_path, dstore_size, dimension, dtype)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.index.nprobe = self.probe
        else:
            start_time = time.time()
            if self.prev_index_path is not None:
                assert os.path.exists(self.trained_index_path), self.trained_index_path
                assert os.path.exists(self.prev_index_path), self.prev_index_path

            if not os.path.exists(self.trained_index_path):
                print ("Sampling...")
                sample_size = 1000000
                random_samples = np.random.choice(np.arange(dstore_size),
                                                  size=[min(sample_size, dstore_size)],
                                                  replace=False)
                t0 = time.time()
                sampled_embs = self.get_embs(random_samples)
                print (time.time()-t0)
                print ("Training index...")
                self._train_index(sampled_embs, self.trained_index_path)
                print ("Finish training (%ds)" % (time.time()-start_time))

            print ("Building index...")
            self.index = self._add_keys(self.index_path, self.prev_index_path if self.prev_index_path is not None else self.trained_index_path)

    def get_embs(self, indices):
        if type(self.embs)==list:
            # indices: [batch_size, K]
            embs = np.zeros((indices.shape[0], indices.shape[1], self.dimension), dtype=self.embs[0].dtype)
            for i, ref_embs in enumerate(self.embs):
                start = self.dstore_size*i
                end = self.dstore_size*(i+1)
                ref_indices = np.minimum(np.maximum(indices, start), end-1)
                embs += (indices >= start) * (indices < self.dstore_size*(i+1)) * ref_embs[ref_indices]
        else:
            embs = self.embs[indices]

        return embs.astype(np.float32)

    def search(self, query_embs, k=4096):
        all_scores, all_indices = self.index.search(query_embs.astype(np.float32), k)
        return all_scores, all_indices

    def get_knn_scores(self, query_emb, indices):
        embs = self.get_embs(indices) # [batch_size, k, dimension]
        scores = - np.sqrt(np.sum((np.expand_dims(query_emb, 1)-embs)**2, -1)) # [batch_size, k]
        return scores

    def _train_index(self, sampled_embs, trained_index_path):
        quantizer = faiss.IndexFlatL2(self.dimension)
        start_index = faiss.IndexIVFPQ(quantizer,
                                       self.dimension,
                                       self.ncentroids,
                                       self.code_size,
                                       8)
        start_index.nprobe = self.probe
        np.random.seed(1)

        if self.cuda:
            # Convert to GPU index
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
            gpu_index.verbose = False

            # Train on GPU and back to CPU
            gpu_index.train(sampled_embs)
            start_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            # Faiss does not handle adding keys in fp16 as of writing this.
            start_index.train(sampled_embs)
        faiss.write_index(start_index, trained_index_path)

    def _add_keys(self, index_path, trained_index_path):
        index = faiss.read_index(trained_index_path)
        start_time = time.time()
        start = 0
        while start < self.dstore_size:
            end = min(self.dstore_size, start + self.num_keys_to_add_at_a_time)
            to_add = self.get_embs(range(start, end)).copy()
            index.add(to_add)
            start = end
            faiss.write_index(index, index_path)

            if start % 5000000 == 0:
                print ('Added %d tokens (%d min)' % (start, (time.time()-start_time)/60))

        print ('Adding took {} s'.format(time.time() - start_time))
        return index
