import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import json
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # Expose WSGI server for cloud deployment

# ── Load saved data ──────────────────────────────────────────────────
df_dash = pd.read_csv('processed_speeches.csv')
eval_dash = pd.read_csv('topic_model_evaluation.csv')
clf_dash = pd.read_csv('classifier_evaluation.csv')

with open('venn_data.json', 'r', encoding='utf-8') as f:
    venn_data = json.load(f)

topic_weight_cols = [c for c in df_dash.columns if c.startswith('Topic_') and c.endswith('_Weight')]
n_topics = len(topic_weight_cols)

# Derive topic labels from data
unique_topics = sorted(df_dash['Dominant_Topic_Label'].unique())

# ── Color palette ────────────────────────────────────────────────────
MODI_COLOR = '#FF6B35'
KHARGE_COLOR = '#004E89'
SPEAKER_COLORS = {'Modi': MODI_COLOR, 'Kharge': KHARGE_COLOR}

# ── Encode Venn diagram image ────────────────────────────────────────
venn_img_b64 = ''
try:
    with open('venn_diagram.png', 'rb') as f:
        venn_img_b64 = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    pass

# ── Pre-build static figures ─────────────────────────────────────────

# 1. Article volume over time
vol_data = df_dash.groupby(['YearMonth', 'Speaker']).size().reset_index(name='Count')
fig_volume = px.bar(
    vol_data, x='YearMonth', y='Count', color='Speaker', barmode='group',
    title='Article Volume Over Time',
    color_discrete_map=SPEAKER_COLORS
)
fig_volume.update_layout(template='plotly_white', xaxis_title='Month', yaxis_title='Articles')

# 2. Overall topic distribution
topic_dist = df_dash.groupby(['Speaker', 'Dominant_Topic_Label']).size().reset_index(name='Count')
topic_dist['Pct'] = topic_dist.groupby('Speaker')['Count'].transform(lambda x: x / x.sum() * 100)
fig_topic_dist = px.bar(
    topic_dist, x='Dominant_Topic_Label', y='Pct', color='Speaker', barmode='group',
    title='Topic Focus by Speaker (% of Articles)',
    color_discrete_map=SPEAKER_COLORS,
    labels={'Pct': '% of Articles', 'Dominant_Topic_Label': 'Topic'}
)
fig_topic_dist.update_layout(template='plotly_white', xaxis_tickangle=-25)

# 3. Model evaluation charts
fig_coherence = px.line(
    eval_dash, x='K', y='Coherence', color='Model', markers=True,
    title='Topic Coherence (UMass) — Higher is Better'
)
fig_coherence.update_layout(template='plotly_white')

fig_silhouette = px.line(
    eval_dash, x='K', y='Silhouette', color='Model', markers=True,
    title='Silhouette Score — Higher is Better'
)
fig_silhouette.update_layout(template='plotly_white')

# 4. Classifier comparison
fig_clf = go.Figure()
fig_clf.add_trace(go.Bar(
    name='CV Accuracy', x=clf_dash['Model'], y=clf_dash['CV_Mean'],
    error_y=dict(type='data', array=clf_dash['CV_Std']),
    marker_color='#2196F3'
))
fig_clf.add_trace(go.Bar(
    name='Test Accuracy', x=clf_dash['Model'], y=clf_dash['Test_Acc'],
    marker_color='#4CAF50'
))
fig_clf.update_layout(
    title='Classifier Performance (CV vs Test Accuracy)',
    barmode='group', template='plotly_white',
    yaxis_title='Accuracy', xaxis_title='Model'
)

# ── Dashboard Layout ─────────────────────────────────────────────────
# Summary stats
total_articles = len(df_dash)
modi_count = len(df_dash[df_dash['Speaker'] == 'Modi'])
kharge_count = len(df_dash[df_dash['Speaker'] == 'Kharge'])

summary_cards = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H4(f"{total_articles}", className='text-center text-primary'),
        html.P('Total Articles', className='text-center text-muted')
    ]), className='shadow-sm'), width=3),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H4(f"{modi_count}", className='text-center', style={'color': MODI_COLOR}),
        html.P('Modi Articles', className='text-center text-muted')
    ]), className='shadow-sm'), width=3),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H4(f"{kharge_count}", className='text-center', style={'color': KHARGE_COLOR}),
        html.P('Kharge Articles', className='text-center text-muted')
    ]), className='shadow-sm'), width=3),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H4(f"{n_topics}", className='text-center text-success'),
        html.P('Topics Identified', className='text-center text-muted')
    ]), className='shadow-sm'), width=3),
], className='mb-4')

# Tabs
tab_overview = dbc.Tab(label='Overview', children=[
    html.Br(),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_volume), width=6),
        dbc.Col(dcc.Graph(figure=fig_topic_dist), width=6),
    ])
])

tab_evolution = dbc.Tab(label='Topic Evolution', children=[
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Label('Select Speaker:', className='fw-bold'),
            dcc.Dropdown(
                id='speaker-dropdown',
                options=[{'label': s, 'value': s} for s in ['Modi', 'Kharge', 'Both']],
                value='Both', clearable=False
            )
        ], width=3),
        dbc.Col([
            html.Label('Chart Type:', className='fw-bold'),
            dcc.Dropdown(
                id='chart-type-dropdown',
                options=[
                    {'label': 'Stacked Area (% Share)', 'value': 'area'},
                    {'label': 'Stacked Bar (Count)', 'value': 'bar'},
                    {'label': 'Line (Weight Trend)', 'value': 'line'}
                ],
                value='area', clearable=False
            )
        ], width=3),
    ], className='mb-3'),
    dcc.Graph(id='evolution-chart')
])

tab_comparison = dbc.Tab(label='Speaker Comparison', children=[
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H5('Topic Word Overlap (Venn Diagram)', className='text-center'),
            html.Img(
                src=f'data:image/png;base64,{venn_img_b64}' if venn_img_b64 else '',
                style={'width': '100%', 'maxWidth': '600px', 'margin': 'auto', 'display': 'block'}
            ) if venn_img_b64 else html.P('Venn diagram not found. Run the Venn cell first.'),
        ], width=6),
        dbc.Col([
            html.H5('Word Set Details', className='text-center'),
            dbc.Card(dbc.CardBody([
                html.H6(f"Modi-Only ({len(venn_data.get('modi_only', []))} terms)", style={'color': MODI_COLOR}),
                html.P(', '.join(venn_data.get('modi_only', [])[:20]), style={'fontSize': '12px'}),
                html.Hr(),
                html.H6(f"Kharge-Only ({len(venn_data.get('kharge_only', []))} terms)", style={'color': KHARGE_COLOR}),
                html.P(', '.join(venn_data.get('kharge_only', [])[:20]), style={'fontSize': '12px'}),
                html.Hr(),
                html.H6(f"Shared ({len(venn_data.get('shared', []))} terms)", style={'color': '#666'}),
                html.P(', '.join(venn_data.get('shared', [])[:20]), style={'fontSize': '12px'}),
            ]), className='shadow-sm')
        ], width=6),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H5('Topic Heatmap: Speaker \u00d7 Topic', className='text-center'),
            dcc.Graph(id='heatmap-chart')
        ])
    ])
])

tab_model = dbc.Tab(label='Model Performance', children=[
    html.Br(),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_coherence), width=6),
        dbc.Col(dcc.Graph(figure=fig_silhouette), width=6),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_clf), width=8),
        dbc.Col([
            html.H5('Classifier Summary'),
            dbc.Table.from_dataframe(
                clf_dash[['Model', 'CV_Mean', 'Test_Acc', 'Overfit_Gap']].round(4),
                striped=True, bordered=True, hover=True, size='sm'
            )
        ], width=4)
    ])
])

project_details = html.Div(style={
    'backgroundColor': '#ffffff', 'borderRadius': '16px', 'padding': '32px', 'marginBottom': '32px',
    'marginTop': '10px', 'boxShadow': '0 10px 25px -5px rgba(0,0,0,0.05)', 'border': '1px solid #e2e8f0',
    'display': 'flex', 'flexWrap': 'wrap', 'gap': '32px', 'alignItems': 'center'
}, children=[
    html.Div(style={'flex': '0 0 auto', 'textAlign': 'center'}, children=[
        html.Img(src='https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg', style={'width': '140px', 'height': 'auto'})
    ]),
    html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
        html.H2('Birla Institute of Technology and Science, Pilani', style={'fontSize': '1.8rem', 'color': '#1e3a8a', 'marginBottom': '4px', 'fontWeight': '700'}),
        html.H3('Pilani Campus', style={'fontSize': '1.2rem', 'color': '#64748b', 'fontWeight': 'bold', 'marginBottom': '16px'}),
        html.P('Submitted By: Group 15', style={'marginBottom': '12px', 'fontWeight': '600', 'color': '#1e293b'}),
        html.Div(style={'overflowX': 'auto', 'backgroundColor': '#f1f5f9', 'borderRadius': '10px', 'padding': '2px'}, children=[
            html.Table(style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': '#fff', 'borderRadius': '8px', 'overflow': 'hidden'}, children=[
                html.Thead(html.Tr([
                    html.Th('Name', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': '1px solid #e2e8f0', 'backgroundColor': '#e2e8f0', 'color': '#334155', 'fontWeight': '600', 'fontSize': '0.9rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                    html.Th('BITS ID', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': '1px solid #e2e8f0', 'backgroundColor': '#e2e8f0', 'color': '#334155', 'fontWeight': '600', 'fontSize': '0.9rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'})
                ])),
                html.Tbody([
                    html.Tr([html.Td('Priyanka Chitlangia', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': '1px solid #e2e8f0', 'fontSize': '0.95rem', 'fontWeight': '500', 'color': '#1e293b'}), html.Td('2025H1540818P', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': '1px solid #e2e8f0', 'fontSize': '0.95rem', 'fontWeight': '500', 'color': '#1e293b'})]),
                    html.Tr([html.Td('Subodh Sanjay Joshi', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': '1px solid #e2e8f0', 'fontSize': '0.95rem', 'fontWeight': '500', 'color': '#1e293b'}), html.Td('2025H1540833P', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': '1px solid #e2e8f0', 'fontSize': '0.95rem', 'fontWeight': '500', 'color': '#1e293b'})]),
                    html.Tr([html.Td('Harsh Jain', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': '1px solid #e2e8f0', 'fontSize': '0.95rem', 'fontWeight': '500', 'color': '#1e293b'}), html.Td('2025H1540837P', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': '1px solid #e2e8f0', 'fontSize': '0.95rem', 'fontWeight': '500', 'color': '#1e293b'})]),
                    html.Tr([html.Td('Aman Tanwar', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': 'none', 'fontSize': '0.95rem', 'fontWeight': '500', 'color': '#1e293b'}), html.Td('2025H1540838P', style={'padding': '12px 20px', 'textAlign': 'left', 'borderBottom': 'none', 'fontSize': '0.95rem', 'fontWeight': '500', 'color': '#1e293b'})])
                ])
            ])
        ])
    ])
])

app.layout = dbc.Container([
    html.H2('Political Speeches — Topic Analysis Dashboard',
            className='text-center my-4', style={'fontWeight': 'bold'}),
    project_details,
    html.P('Comparing discourse themes of Modi and Kharge across time using NLP topic modeling.',
           className='text-center text-muted mb-4'),
    summary_cards,
    dbc.Tabs([tab_overview, tab_evolution, tab_comparison, tab_model])
], fluid=True)

# ── Callbacks ─────────────────────────────────────────────────────────

@app.callback(
    Output('evolution-chart', 'figure'),
    [Input('speaker-dropdown', 'value'),
     Input('chart-type-dropdown', 'value')]
)
def update_evolution(speaker, chart_type):
    if speaker == 'Both':
        data = df_dash.copy()
    else:
        data = df_dash[df_dash['Speaker'] == speaker].copy()
    
    if chart_type == 'area':
        tw = data.groupby(['YearMonth', 'Speaker'])[topic_weight_cols].mean().reset_index()
        tw_melted = tw.melt(
            id_vars=['YearMonth', 'Speaker'], value_vars=topic_weight_cols,
            var_name='Topic', value_name='Weight'
        )
        tw_melted['Topic'] = tw_melted['Topic'].apply(
            lambda x: unique_topics[int(x.split('_')[1])] if int(x.split('_')[1]) < len(unique_topics) else x
        )
        
        if speaker == 'Both':
            fig = px.area(
                tw_melted, x='YearMonth', y='Weight', color='Topic',
                facet_col='Speaker', groupnorm='percent',
                title='Topic Evolution (Proportional Share)'
            )
        else:
            fig = px.area(
                tw_melted, x='YearMonth', y='Weight', color='Topic',
                groupnorm='percent',
                title=f'{speaker} — Topic Evolution (Proportional Share)'
            )
    elif chart_type == 'bar':
        counts = data.groupby(['YearMonth', 'Speaker', 'Dominant_Topic_Label']).size().reset_index(name='Count')
        if speaker == 'Both':
            fig = px.bar(
                counts, x='YearMonth', y='Count', color='Dominant_Topic_Label',
                facet_col='Speaker', barmode='stack',
                title='Topic Counts Over Time (Stacked)'
            )
        else:
            fig = px.bar(
                counts, x='YearMonth', y='Count', color='Dominant_Topic_Label',
                barmode='stack',
                title=f'{speaker} — Topic Counts Over Time'
            )
    else:  # line
        tw = data.groupby(['YearMonth', 'Speaker'])[topic_weight_cols].mean().reset_index()
        tw_melted = tw.melt(
            id_vars=['YearMonth', 'Speaker'], value_vars=topic_weight_cols,
            var_name='Topic', value_name='Weight'
        )
        tw_melted['Topic'] = tw_melted['Topic'].apply(
            lambda x: unique_topics[int(x.split('_')[1])] if int(x.split('_')[1]) < len(unique_topics) else x
        )
        
        if speaker == 'Both':
            fig = px.line(
                tw_melted, x='YearMonth', y='Weight', color='Topic',
                facet_col='Speaker', markers=True,
                title='Topic Weight Trends'
            )
        else:
            fig = px.line(
                tw_melted, x='YearMonth', y='Weight', color='Topic',
                markers=True,
                title=f'{speaker} — Topic Weight Trends'
            )
    
    fig.update_layout(template='plotly_white', height=500)
    return fig


@app.callback(
    Output('heatmap-chart', 'figure'),
    Input('speaker-dropdown', 'value')
)
def update_heatmap(_):
    heatmap_data = df_dash.groupby('Speaker')[topic_weight_cols].mean()
    heatmap_data.columns = [unique_topics[int(c.split('_')[1])] if int(c.split('_')[1]) < len(unique_topics) else c for c in heatmap_data.columns]
    
    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        color_continuous_scale='RdYlBu_r',
        title='Average Topic Affinity by Speaker',
        labels={'color': 'Avg Weight'}
    )
    fig.update_layout(template='plotly_white', height=350)
    return fig

# ── Run ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print(f"Starting dashboard on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
