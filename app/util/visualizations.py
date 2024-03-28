import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots


def prepare_data(vectors, labels, vectorizer):
    email_data = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())
    email_data['label'] = labels
    email_data['label'] = email_data['label'].apply(lambda x: 'Phishing' if x == 1 else 'Legitimate')
    aggregated_data = email_data.groupby('label').mean()
    return email_data, aggregated_data


def plot_html_proportion(email_data):
    # Calculate the proportion of emails that contain HTML content for each label
    html_proportion = email_data.groupby('label')['has_html'].mean() * 100  # assuming 'has_html' is a binary feature

    # Convert the Series to DataFrame for better control over plotting
    html_proportion_df = html_proportion.reset_index()
    html_proportion_df.columns = ['Email Category', 'Proportion (%)']

    # Plotting
    fig = px.bar(html_proportion_df, x='Email Category', y='Proportion (%)',
                 title='Proportion of Emails containing HTML by Category',
                 color='Email Category',
                 color_discrete_map={'Legitimate': 'blue', 'Phishing': 'red'})

    fig.update_layout(xaxis_title='Email Category', yaxis_title='Proportion (%)', yaxis=dict(tickformat=".2f"),
                      showlegend=False)

    return fig.to_html()


def plot_body_length_distribution(email_data):
    fig_length = px.histogram(email_data, x='body_length', color='label', title='Distribution of Email Body Length',
                              color_discrete_map={'Legitimate': 'blue', 'Phishing': 'red'}, labels={'label': ''})
    fig_length.update_xaxes(title_text='Email Body Length', range=[0, 20000])
    fig_length.update_yaxes(title_text='Count')
    return fig_length.to_html()


def plot_links_pie_chart(email_data):
    link_counts = email_data.groupby(['label', 'has_links']).size().unstack(fill_value=0)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                        subplot_titles=['Legitimate Emails', 'Phishing Emails'])

    for i, label in enumerate(link_counts.index):
        values = link_counts.loc[label]
        fig.add_trace(go.Pie(labels=['No Links', 'Has Links'], values=values, name=label), row=1, col=i + 1)

    fig.update_traces(marker=dict(colors=['blue', 'red']))
    fig.update_layout(title_text='Proportion of Emails with and without Links')

    return fig.to_html()


def plot_sensitive_info_comparison(email_data):
    # Filter emails with sensitive information
    sensitive_info_emails = email_data[email_data['has_sensitive_info'] == 1.0]

    # Get the proportion of emails with sensitive information by label
    total_counts = email_data['label'].value_counts()
    sensitive_counts = sensitive_info_emails['label'].value_counts()
    proportions = (sensitive_counts / total_counts).reset_index()
    proportions.columns = ['label', 'proportion']

    # Create the bar chart for proportions
    fig = px.bar(proportions, x='label', y='proportion',
                 title='Proportional Comparison of Emails Requesting Sensitive Information',
                 color='label', color_discrete_map={'Legitimate': 'blue', 'Phishing': 'red'},
                 labels={'label': '', 'proportion': 'Proportion of Emails'})

    fig.update_xaxes(title_text='Email Type')
    fig.update_yaxes(title_text='Proportion of Emails Requesting Sensitive Info')

    return fig.to_html()

