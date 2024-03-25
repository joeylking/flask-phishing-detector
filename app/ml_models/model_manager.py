import pickle
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelManager:
    def __init__(self):


    def train_and_evaluate(self, model, name, train_vectors, train_labels, test_vectors, test_labels):
        training_start_time = time.time()
        print(f"Training {name}...")
        dense_required = ['G Naive Bayes', 'CatBoost', 'Hist Gradient Boosting']

        if name in dense_required:
            # Convert to dense format for models that prefer dense data
            train_vectors_dense = train_vectors.toarray()
            test_vectors_dense = test_vectors.toarray()
            model.fit(train_vectors_dense, train_labels)
            predicted_labels = model.predict(test_vectors_dense)
        else:
            model.fit(train_vectors, train_labels)
            predicted_labels = model.predict(test_vectors)

        training_end_time = time.time()
        print(f"Training time: {training_end_time - training_start_time}")

        print(f"Evaluating {name}...")
        accuracy = accuracy_score(test_labels, predicted_labels)
        precision = precision_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels)

        print(f"{name} accuracy: {accuracy:.3f}")
        print(f"{name} precision: {precision:.3f}")
        print(f"{name} recall: {recall:.3f}")
        print(f"{name} F1 score: {f1:.3f}")

        # Save the trained model
        model_filename = f"{name.replace(' ', '_')}_model.pkl"
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Saved {name} model to {model_filename}")