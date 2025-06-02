"""
Visualization functions for MCDM results
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from config.settings import VIZ_CONFIG

def create_results_chart(results_df, method_name):
    """
    Create a bar chart showing alternative scores
    
    Args:
        results_df (pd.DataFrame): Results dataframe with Alternative, Score, Rank columns
        method_name (str): Name of the MCDM method
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Sort by rank for better visualization
    df_sorted = results_df.sort_values('Rank')
    
    # Create color scale based on rank (best = green, worst = red)
    colors = px.colors.sample_colorscale(
        'RdYlGn_r', 
        [i/(len(df_sorted)-1) for i in range(len(df_sorted))]
    )
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_sorted['Alternative'],
            y=df_sorted['Score'],
            text=[f"Rank {rank}" for rank in df_sorted['Rank']],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>' +
                         'Score: %{y:.4f}<br>' +
                         'Rank: %{text}<br>' +
                         '<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'{method_name} Method - Alternative Scores',
        xaxis_title='Alternatives',
        yaxis_title='Score',
        height=VIZ_CONFIG['chart_height'],
        showlegend=False,
        template='plotly_white'
    )
    
    # Add rank annotations
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        fig.add_annotation(
            x=row['Alternative'],
            y=row['Score'],
            text=f"#{row['Rank']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black",
            ax=0,
            ay=-30
        )
    
    return fig

def create_comparison_chart(decision_matrix, criteria, alternatives_to_show=None):
    """
    Create a radar chart comparing alternatives across criteria
    
    Args:
        decision_matrix (pd.DataFrame): Decision matrix
        criteria (list): List of criteria names
        alternatives_to_show (list, optional): Specific alternatives to show
        
    Returns:
        plotly.graph_objects.Figure: Radar chart figure
    """
    if alternatives_to_show is None:
        alternatives_to_show = decision_matrix.index.tolist()
    
    # Normalize data for radar chart (0-1 scale)
    normalized_matrix = decision_matrix.copy()
    for col in normalized_matrix.columns:
        col_min = normalized_matrix[col].min()
        col_max = normalized_matrix[col].max()
        if col_max != col_min:
            normalized_matrix[col] = (normalized_matrix[col] - col_min) / (col_max - col_min)
        else:
            normalized_matrix[col] = 0.5  # If all values are the same
    
    fig = go.Figure()
    
    colors = VIZ_CONFIG['color_palette']
    
    for i, alternative in enumerate(alternatives_to_show):
        if alternative in normalized_matrix.index:
            values = normalized_matrix.loc[alternative].tolist()
            values.append(values[0])  # Close the radar chart
            
            criteria_extended = criteria + [criteria[0]]  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=criteria_extended,
                fill='toself',
                name=alternative,
                line_color=colors[i % len(colors)],
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             '%{theta}: %{r:.3f}<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Multi-Criteria Comparison (Normalized Values)",
        height=VIZ_CONFIG['chart_height'],
        template='plotly_white'
    )
    
    return fig

def create_weights_chart(criteria, weights):
    """
    Create a pie chart showing criteria weights
    
    Args:
        criteria (list): List of criteria names
        weights (list): List of weight values
        
    Returns:
        plotly.graph_objects.Figure: Pie chart figure
    """
    fig = go.Figure(data=[go.Pie(
        labels=criteria,
        values=weights,
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>' +
                     'Weight: %{value:.3f}<br>' +
                     'Percentage: %{percent}<br>' +
                     '<extra></extra>'
    )])
    
    fig.update_layout(
        title="Criteria Weights Distribution",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_sensitivity_chart(base_scores, sensitivity_data, alternative_names):
    """
    Create a line chart showing sensitivity analysis results
    
    Args:
        base_scores (list): Original scores
        sensitivity_data (dict): Dictionary with weight changes and resulting scores
        alternative_names (list): Names of alternatives
        
    Returns:
        plotly.graph_objects.Figure: Line chart figure
    """
    fig = go.Figure()
    
    colors = VIZ_CONFIG['color_palette']
    
    # Add base scores
    fig.add_trace(go.Scatter(
        x=[0],
        y=base_scores,
        mode='markers',
        name='Base Case',
        marker=dict(size=10, color='black'),
        hovertemplate='Base Case<br>Score: %{y:.4f}<extra></extra>'
    ))
    
    # Add sensitivity lines for each alternative
    for i, alt_name in enumerate(alternative_names):
        x_values = list(sensitivity_data.keys())
        y_values = [scores[i] for scores in sensitivity_data.values()]
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=alt_name,
            line=dict(color=colors[i % len(colors)]),
            hovertemplate=f'<b>{alt_name}</b><br>' +
                         'Weight Change: %{x}<br>' +
                         'Score: %{y:.4f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Sensitivity Analysis - Score Changes with Weight Variations",
        xaxis_title="Weight Change (%)",
        yaxis_title="Alternative Scores",
        height=VIZ_CONFIG['chart_height'],
        template='plotly_white'
    )
    
    return fig

def create_ranking_stability_chart(ranking_data, alternative_names):
    """
    Create a chart showing ranking stability across different scenarios
    
    Args:
        ranking_data (dict): Dictionary with scenarios and rankings
        alternative_names (list): Names of alternatives
        
    Returns:
        plotly.graph_objects.Figure: Heatmap figure
    """
    # Create matrix for heatmap
    scenarios = list(ranking_data.keys())
    rankings_matrix = []
    
    for scenario in scenarios:
        rankings_matrix.append(ranking_data[scenario])
    
    fig = go.Figure(data=go.Heatmap(
        z=rankings_matrix,
        x=alternative_names,
        y=scenarios,
        colorscale='RdYlGn_r',
        text=rankings_matrix,
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate='<b>%{y}</b><br>' +
                     'Alternative: %{x}<br>' +
                     'Rank: %{z}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Ranking Stability Across Scenarios",
        xaxis_title="Alternatives",
        yaxis_title="Scenarios",
        height=VIZ_CONFIG['chart_height'],
        template='plotly_white'
    )
    
    return fig

def create_topsis_analysis_chart(method_instance):
    """
    Create a specialized chart for TOPSIS showing distances to ideal solutions

    Args:
        method_instance: TOPSIS method instance with calculated results

    Returns:
        plotly.graph_objects.Figure: TOPSIS analysis chart
    """
    if not hasattr(method_instance, 'intermediate_steps'):
        return None

    steps = method_instance.intermediate_steps
    alternatives = method_instance.alternatives

    # Create subplot with secondary y-axis
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distance Analysis', 'Relative Closeness'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Distance analysis (left subplot)
    fig.add_trace(
        go.Bar(
            x=alternatives,
            y=steps['s_plus'],
            name='Distance to PIS (S+)',
            marker_color='lightcoral',
            text=[f"{val:.4f}" for val in steps['s_plus']],
            textposition='auto'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=alternatives,
            y=steps['s_minus'],
            name='Distance to NIS (S-)',
            marker_color='lightblue',
            text=[f"{val:.4f}" for val in steps['s_minus']],
            textposition='auto'
        ),
        row=1, col=1
    )

    # Relative closeness (right subplot)
    colors = ['gold' if rank == 1 else 'lightgreen' if rank <= 3 else 'lightgray'
              for rank in method_instance.results['rankings']]

    fig.add_trace(
        go.Bar(
            x=alternatives,
            y=steps['relative_closeness'],
            name='Relative Closeness (C*)',
            marker_color=colors,
            text=[f"{val:.4f}" for val in steps['relative_closeness']],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="TOPSIS Analysis: Distance Measures and Relative Closeness",
        height=500,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Alternatives", row=1, col=1)
    fig.update_xaxes(title_text="Alternatives", row=1, col=2)
    fig.update_yaxes(title_text="Distance", row=1, col=1)
    fig.update_yaxes(title_text="Relative Closeness", row=1, col=2)

    return fig

def create_method_comparison_chart(results_dict):
    """
    Create a chart comparing results from different MCDM methods

    Args:
        results_dict (dict): Dictionary with method names as keys and results as values

    Returns:
        plotly.graph_objects.Figure: Method comparison chart
    """
    if not results_dict or len(results_dict) < 2:
        return None

    # Get alternatives from first method
    first_method = list(results_dict.values())[0]
    alternatives = first_method['alternatives'] if 'alternatives' in first_method else []

    fig = go.Figure()

    colors = VIZ_CONFIG['color_palette']

    for i, (method_name, results) in enumerate(results_dict.items()):
        rankings = results.get('rankings', [])

        fig.add_trace(go.Scatter(
            x=alternatives,
            y=rankings,
            mode='lines+markers',
            name=method_name,
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=8),
            hovertemplate=f'<b>{method_name}</b><br>' +
                         'Alternative: %{x}<br>' +
                         'Rank: %{y}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title="Method Comparison: Rankings Across Different MCDM Methods",
        xaxis_title="Alternatives",
        yaxis_title="Rank (1 = Best)",
        yaxis=dict(autorange="reversed"),  # Reverse y-axis so rank 1 is at top
        height=VIZ_CONFIG['chart_height'],
        template='plotly_white',
        hovermode='x unified'
    )

    return fig

def create_criteria_impact_chart(decision_matrix, weights, criterion_types):
    """
    Create a chart showing the impact of each criterion

    Args:
        decision_matrix (pd.DataFrame): Decision matrix
        weights (list): Criteria weights
        criterion_types (list): Criterion types (benefit/cost)

    Returns:
        plotly.graph_objects.Figure: Criteria impact chart
    """
    criteria = decision_matrix.columns.tolist()

    # Calculate coefficient of variation for each criterion (measure of variability)
    cv_values = []
    for col in criteria:
        mean_val = decision_matrix[col].mean()
        std_val = decision_matrix[col].std()
        cv = std_val / mean_val if mean_val != 0 else 0
        cv_values.append(cv)

    # Calculate weighted impact (weight * coefficient of variation)
    weighted_impact = [w * cv for w, cv in zip(weights, cv_values)]

    # Create subplot
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Criteria Weights', 'Criteria Impact (Weight × Variability)'),
        vertical_spacing=0.15
    )

    # Weights chart
    colors_weights = ['lightcoral' if ct == 'cost' else 'lightblue' for ct in criterion_types]

    fig.add_trace(
        go.Bar(
            x=criteria,
            y=weights,
            name='Weights',
            marker_color=colors_weights,
            text=[f"{w:.3f}" for w in weights],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )

    # Impact chart
    fig.add_trace(
        go.Bar(
            x=criteria,
            y=weighted_impact,
            name='Weighted Impact',
            marker_color='gold',
            text=[f"{wi:.3f}" for wi in weighted_impact],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="Criteria Analysis: Weights and Impact on Decision",
        height=600,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Criteria", row=2, col=1)
    fig.update_yaxes(title_text="Weight", row=1, col=1)
    fig.update_yaxes(title_text="Impact Score", row=2, col=1)

    return fig
