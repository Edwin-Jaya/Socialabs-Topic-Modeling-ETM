from octis.dataset.dataset import Dataset
from octis.models.ETM import ETM
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
import pandas
import numpy


class EmbeddedTopicModeling:
    def __init__(self, dataset_path, original_df):
        self.dataset = Dataset()
        self.dataset.load_custom_dataset_from_folder(dataset_path)
        self.original_df = original_df
        self.model =self.train_model(self.dataset)

    def train_model(self, dataset):
        best_coh = float("-inf")
        best_topic = None
        best_model = None
        best_model_output = None

        coh_score_list = []
        topics = range(4, 8)

        for topic in topics:
            model = ETM(
                num_topics=topic,
                num_epochs=100,
                batch_size=256,
                dropout=0.2,
                embedding_size=100,
                t_hidden_size=256,
                wdecay=1e-9,
                lr=0.002,
                optimizer='adam'
            )

            # train
            model_output = model.train_model(dataset)
            print(f"[{topic} topics] Model trained.")

            # evaluate
            coh_score = self.evaluate_coherence(dataset, model_output)
            coh_score_list.append(coh_score)
            print(f"[{topic} topics] Coherence = {coh_score:.4f}")

            # check if this is the best so far
            if coh_score > best_coh:
                best_coh = coh_score
                best_topic = topic
                best_model = model
                best_model_output = model_output

        # after looping
        print("\n=== Summary ===")
        for t, score in zip(topics, coh_score_list):
            print(f"  Topics={t} → Coherence={score:.4f}")

        print(f"\nBest model has {best_topic} topics with coherence={best_coh:.4f}")

        matrix = model_output['topic-document-matrix']
        # Mencari indeks baris dengan nilai maksimum untuk setiap kolom
        max_indices = numpy.argmax(matrix, axis=0)

        # Konversi ke 1-based dan buat list hasil
        result = (max_indices + 1).tolist()

        print("Hasil list dengan nilai tertinggi setiap kolom:")
        
        train_corpus = dataset.get_partitioned_corpus()[0]
        df = pandas.DataFrame(columns=["id_str", "dominant_topic", "tweets"])
        df["id_str"] = self.original_df["id_str"][:len(train_corpus)]
        df["dominant_topic"] = result
        df["tweets"] = train_corpus

        return df

    def evaluate_coherence(self, dataset, model_output):
        coh = Coherence(texts=dataset.get_corpus(),topk=10,
                    measure='c_v')
        
        return coh.score(model_output)