from src.models.kmeans import KMeansClustering
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.load_data import load_data

def tune_model(data, model, params):
    pipeline = PreprocessingPipeline(data, model, params)
    pipeline.fit_transform()
    return pipeline

if __name__ == "__main__":
    data = load_data()
    model = KMeansClustering()
    params = {
        'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10]}