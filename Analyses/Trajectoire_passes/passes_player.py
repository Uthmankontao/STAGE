import plotly.graph_objects as go
import pandas as pd
    
def Joueur_passes(df, team_name, period, player_name):
    """Pour voir les passes d'un joueur spécifique au cours de la premiere oi la deuxieme periode"""
    fig = go.Figure()

    terrain_shapes = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="Black")),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="Black", dash="dot")),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="Black")),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="Black")),
        dict(type="rect", x0=0, y0=30, x1=5.5, y1=50, line=dict(color="Black")),
        dict(type="rect", x0=114.5, y0=30, x1=120, y1=50, line=dict(color="Black")),
        dict(type="rect", x0=0, y0=36, x1=2, y1=44, line=dict(color="Black")),
        dict(type="rect", x0=118, y0=36, x1=120, y1=44, line=dict(color="Black")),
        dict(type="circle", xref="x", yref="y", x0=50.4, y0=30.4, x1=69.6, y1=49.6, line_color="Black"),
    ]
    fig.update_layout(shapes=terrain_shapes)

    df = df[(df['team.name'] == team_name) & (df['period'] == period) &
            (df['player.name'] == player_name) &
            df['location.x'].notna() & df['location.y'].notna() &
            df['pass.end_location_x'].notna() & df['pass.end_location_y'].notna()]

    for _, pass_ in df.iterrows():
        if period == 2:
            start_x = round(120 - pass_['location.x'], 2)
            end_x = round(120 - pass_['pass.end_location_x'], 2)
            start_y = round(80 - pass_['location.y'], 2)
            end_y = round(80 - pass_['pass.end_location_y'], 2)
        else:
            start_x, end_x = pass_['location.x'], pass_['pass.end_location_x']
            start_y, end_y = pass_['location.y'], pass_['pass.end_location_y']

        fig.add_trace(go.Scatter(
            x=[start_x, end_x],
            y=[start_y, end_y],
            mode='lines+markers',
            name=f"{pass_['player.name']} to {pass_['pass.recipient.name']}",
            marker=dict(size=5),
            line=dict(width=2, color='purple'),
            hoverinfo='text',
            text=f"De: {pass_['player.name']}<br>À: {pass_['pass.recipient.name']}<br>X: {start_x} -> {end_x}<br>Y: {start_y} -> {end_y}"
        ))

        fig.add_annotation(
            x=end_x,
            y=end_y,
            ax=start_x,
            ay=start_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="purple"
        )

    fig.update_yaxes(autorange='reversed')
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, range=[0, 120]),
        yaxis=dict(showgrid=False, zeroline=False, range=[0, 80]),
        title=f"Les passes de {player_name} durant la {period} période de jeu",
        width=1000,
        height=700
    )

    fig.show()
    