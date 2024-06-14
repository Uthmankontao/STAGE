import plotly.graph_objects as go

def Tirs(df, home_team, away_team, selected_period):
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
    df = df[(df['period'] == selected_period) & df['location.x'].notna() & df['location.y'].notna() & df['shot.end_location.x'].notna() & df['shot.end_location.y'].notna()]

    color_map = {
        "Goal": "green",
        "Saved": "blue",
        "Missed": "red",
        "Blocked": "orange",
        "Off T": "purple" 
    }

    for i, tir in df.iterrows():
        if (tir['team.name'] == home_team and tir['period'] == 1) or (tir['team.name'] == away_team and tir['period'] == 2):
            start_x, end_x = tir['location.x'], tir['shot.end_location.x']
            start_y, end_y = tir['location.y'], tir['shot.end_location.y']
        else:
            start_x, end_x = 120 - tir['location.x'], 120 - tir['shot.end_location.x']
            start_y, end_y = 80 - tir['location.y'], 80 - tir['shot.end_location.y']

        arrow_color = color_map.get(tir['shot.outcome.name'], "gray") 

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
            arrowcolor=arrow_color
        )

        hovertext = f"{tir['player.name']} ({tir['team.name']})<br>Resultat: {tir['shot.outcome.name']}<br>De: ({tir['location.x']}, {tir['location.y']})<br>A: ({tir['shot.end_location.x']}, {tir['shot.end_location.y']})"

        fig.add_trace(go.Scatter(
            x=[start_x, end_x],
            y=[start_y, end_y],
            mode='lines+markers',
            name=f"{tir['player.name']} ({tir['team.name']}) - {tir['shot.outcome.name']}",
            marker=dict(size=10, symbol='circle'),
            line=dict(width=2, color=arrow_color),
            hoverinfo='text',
            text=hovertext
        ))

    fig.update_yaxes(autorange='reversed')
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, range=[0, 120]),
        yaxis=dict(showgrid=False, zeroline=False, range=[0, 80]),
        width=1000,
        height=700,
        title=f"Tirs du match {home_team} contre {away_team} - Pendant la {selected_period} période de jeu"
    )

    fig.show()