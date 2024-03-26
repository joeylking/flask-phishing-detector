from flask import Blueprint
# Import necessary functions for data loading

data_loading_blueprint = Blueprint('data_loading', __name__)

@data_loading_blueprint.route('/load')
def load_data():
    # Load and preprocess the data
    return "Data loaded successfully"
