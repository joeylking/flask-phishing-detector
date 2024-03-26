import logging
import os
import pickle

from flask import Flask
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from app.features.feature_extractor import FeatureExtractor
from app.ml_models.model_manager import ModelManager
from app.routes import *
from app.routes.data_loading import data_loading_blueprint
from app.routes.eda import eda_blueprint
from app.routes.home import home_blueprint
from app.routes.prediction import prediction_blueprint
from app.util.email_loading_utils import load_enron_dataset, load_nazario_phishing_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)


def create_app():
    app = Flask(__name__)

    # Initial setup
    initialize_app()

    app.register_blueprint(home_blueprint)
    app.register_blueprint(eda_blueprint, url_prefix='/eda')
    app.register_blueprint(prediction_blueprint, url_prefix='/predict')
    app.register_blueprint(data_loading_blueprint, url_prefix='/data')

    return app


def initialize_app():
    logging.info("Starting application...")

    # Define file paths
    train_vectors_file = 'app/ml_models/train_vectors.pkl'
    train_labels_file = 'app/ml_models/train_labels.pkl'
    test_vectors_file = 'app/ml_models/test_vectors.pkl'
    test_labels_file = 'app/ml_models/test_labels.pkl'

    # Check for existing data files and load them if they exist
    if os.path.exists(train_vectors_file) and os.path.exists(train_labels_file) and \
            os.path.exists(test_vectors_file) and os.path.exists(test_labels_file):
        logging.info("Loading preprocessed data from pickle files...")
        # Load pickle files
        with open(train_vectors_file, 'rb') as f:
            train_vectors = pickle.load(f)
        with open(test_vectors_file, 'rb') as f:
            test_vectors = pickle.load(f)
        with open(train_labels_file, 'rb') as f:
            train_labels = pickle.load(f)
        with open(test_labels_file, 'rb') as f:
            test_labels = pickle.load(f)
        logging.info("Data loading complete.")
    else:
        logging.info("Preprocessed data not found, loading and processing raw datasets...")

        # Load emails
        enron_emails = load_enron_dataset('app/data/enron-data/maildir', ['inbox'])
        nazario_emails = load_nazario_phishing_dataset('app/data/phishing-data/')
        print("All emails loaded")

        # Apply labels
        enron_labels = [0] * len(enron_emails)  # 0 for regular
        nazario_labels = [1] * len(nazario_emails)  # 1 for phishing

        # Split the datasets into training and testing sets
        enron_train, enron_test, enron_train_labels, enron_test_labels = train_test_split(enron_emails, enron_labels,
                                                                                          test_size=0.20,
                                                                                          random_state=42)
        nazario_train, nazario_test, nazario_train_labels, nazario_test_labels = train_test_split(nazario_emails,
                                                                                                  nazario_labels,
                                                                                                  test_size=0.20,
                                                                                                  random_state=42)
        # Combine the training and testing sets
        train_emails = enron_train + nazario_train
        test_emails = enron_test + nazario_test
        train_labels = enron_train_labels + nazario_train_labels
        test_labels = enron_test_labels + nazario_test_labels

        # Extract features for training and testing sets
        logging.info("Extracting features...")
        extractor = FeatureExtractor()
        train_features = [extractor.extract_features_with_logging(train_emails.index(email), "training", email)
                          for email in train_emails]
        test_features = [extractor.extract_features_with_logging(test_emails.index(email), "testing", email)
                         for email in test_emails]

        # Convert the list of feature dicts to a feature matrix
        logging.info("Begin vectorization...")
        vectorizer = DictVectorizer(sparse=True)
        train_vectors = vectorizer.fit_transform(train_features)
        test_vectors = vectorizer.transform(test_features)

        # Save vectors and labels
        logging.info("Saving vectors and labels...")
        with open(train_vectors_file, 'wb') as f:
            pickle.dump(train_vectors, f)
        with open(test_vectors_file, 'wb') as f:
            pickle.dump(test_vectors, f)
        with open(train_labels_file, 'wb') as f:
            pickle.dump(train_labels, f)
        with open(test_labels_file, 'wb') as f:
            pickle.dump(test_labels, f)

        logging.info("Data processing complete.")

    # Model training and loading
    # Define file path
    rf_model_file = 'app/ml_models/Random_Forest_model.pkl'
    manager = ModelManager()
    if os.path.exists(rf_model_file):
        logging.info("Random Forest model already trained.")
        with open(rf_model_file, 'rb') as f:
            model = pickle.load(f)
        predicted_labels = model.predict(test_vectors)
        manager.evaluate_model(model, "Random Forest", predicted_labels, test_labels)
    else:
        logging.info("Training new Random Forest model...")
        manager.train_and_evaluate(make_pipeline(SimpleImputer(strategy="mean"),
                                                 RandomForestClassifier(n_estimators=100)),
                                   'Random Forest', train_vectors, train_labels, test_vectors, test_labels)



    logging.info("Model setup complete.")
