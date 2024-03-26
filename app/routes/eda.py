import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Blueprint, send_file, render_template, url_for
import io

from app.util.email_loading_utils import load_enron_dataset, load_nazario_phishing_dataset

eda_blueprint = Blueprint('eda', __name__)

@eda_blueprint.route('/')
def show_eda():
    enron_data = load_enron_dataset('app/data/enron-data/maildir', sample_fraction=0.1)
    nazario_data = load_nazario_phishing_dataset('app/data/phishing-data/', sample_fraction=0.1)

    # Combine and label the datasets
    enron_df = pd.DataFrame(enron_data)
    enron_df['label'] = 'Regular'
    nazario_df = pd.DataFrame(nazario_data)
    nazario_df['label'] = 'Phishing'
    combined_df = pd.concat([enron_df, nazario_df])

    # Perform EDA
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    sns.histplot(data=combined_df, x='subject_length', hue='label', element='step')
    plt.title('Distribution of Email Subject Lengths')
    plot_url = url_for('static', filename='eda_plot.png')
    plt.savefig(f'./static/eda_plot.png')
    return render_template('eda.html')
