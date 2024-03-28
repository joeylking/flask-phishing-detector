from flask import Blueprint, render_template, request, current_app

from app import FeatureExtractor
from app.util.visualizations import prepare_data, plot_body_length_distribution, plot_html_proportion, \
    plot_links_pie_chart, plot_sensitive_info_comparison

home_blueprint = Blueprint('home', __name__, template_folder='../templates')

feature_extractor = FeatureExtractor()


@home_blueprint.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email_text = request.form.get('email_text')
        email_file = request.files.get('email_file')

        if email_text:
            email_data = email_text
        elif email_file:
            email_data = email_file.read().decode('utf-8')
        else:
            return render_template('index.html', error='No email data provided')

        features = feature_extractor.extract(email_data)
        features_vector = current_app.config['VECTORIZER'].transform([features])
        prediction = current_app.config['MODEL'].predict(features_vector)
        confidence = max(current_app.config['MODEL'].predict_proba(features_vector)[0] * 100)

        # For simplicity, assuming binary classification: 0 for legitimate, 1 for phishing
        result = 'Phishing' if prediction[0] == 1 else 'Legitimate'

        return render_template('index.html', result=result, confidence=confidence)

    return render_template('index.html')


@home_blueprint.route('/dashboard')
def dashboard():
    email_data, aggregated_data = prepare_data(current_app.config['TRAIN_VECTOR'], current_app.config['TRAIN_LABELS'],
                                               current_app.config['VECTORIZER'])

    html_proportion_graph = plot_html_proportion(email_data)
    body_length_distribution_graph = plot_body_length_distribution(email_data)
    links_pie_chart = plot_links_pie_chart(email_data)
    sensitive_info_comparison = plot_sensitive_info_comparison(email_data)

    return render_template('dashboard.html', html_proportion_graph=html_proportion_graph,
                           links_pie_chart=links_pie_chart,
                           body_length_distribution_graph=body_length_distribution_graph,
                           sensitive_info_comparison=sensitive_info_comparison)
