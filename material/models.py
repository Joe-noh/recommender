from django.conf import settings
from django.db import models
from surprise import Dataset, Reader, NMF
from os import path

# plain python class
class UserBasedMaterialRecommender:
    alg = NMF()
    data = None
    trainset = None

    @classmethod
    def fit(klass):
        data_path = path.join(settings.BASE_DIR, 'dataset.csv')
        reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 2))

        klass.data = Dataset.load_from_file(data_path, reader)
        klass.trainset = klass.data.build_full_trainset()
        klass.alg.fit(klass.trainset)

    @classmethod
    def predict(klass, user_id):
        proc = lambda iid: klass.alg.predict(str(user_id), klass.trainset.to_raw_iid(iid))
        return list(map(proc, klass.trainset.all_items()))
