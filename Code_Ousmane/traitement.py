import numpy as np
import pandas as pd
import plotly.graph_objects as go

def filtrer_et_formater_passes(events):
    # Définir les colonnes d'intérêt
    passes = ['datetime', 'team.name', 'period', 'player.name', 'pass.recipient.name', 'pass.length']
    
    
    df_passes = events[passes].copy()
    
    # Filtrer les lignes où 'player.name' et 'pass.recipient.name' ne sont pas nulles
    df_passes = df_passes[df_passes['player.name'].notna() & df_passes['pass.recipient.name'].notna()]
    
    # Réinitialiser l'index et ajouter 1 pour commencer à partir de 1
    df_passes.reset_index(drop=True, inplace=True)
    df_passes.index = df_passes.index + 1
    
    return df_passes

def sauvegarder_dataframe_en_csv(df, nom_du_fichier_sortie):
    df.to_csv(nom_du_fichier_sortie, index=False)
    print(f"DataFrame sauvegardé en tant que {nom_du_fichier_sortie}")

def nettoyer_et_formatter_dataframe(nom_du_fichier):
    # Charger le fichier CSV
    df = pd.read_csv(nom_du_fichier)
    
    # Définir les nouveaux noms de colonnes selon la documentation
    nouveaux_noms_de_colonnes = [
        'Numéro de maillot', 'Type d\'objet', 'Prénom', 'Nom de famille', 'ID de position',
        'Date de naissance du joueur', 'Temps entré', 'Temps sorti', 'ID de joueur événement',
        'ID de joueur suivi', 'ID de match événement', 'ID de match suivi'
    ]
    
    # Renommer les colonnes
    df.columns = nouveaux_noms_de_colonnes
    
    # Convertir la colonne "ID de match suivi" en chaîne de caractères et supprimer le point-virgule
    df['ID de match suivi'] = df['ID de match suivi'].astype(str).str.replace(';', '')



    
    # Supprimer les colonnes non nécessaires
    df = df.drop(columns=['Numéro de maillot', "Type d'objet", 'ID de position',
                          'Date de naissance du joueur', 'ID de joueur événement',
                          'ID de joueur suivi', 'ID de match événement', 'ID de match suivi'])
    
def get_ball_data(filename):
    
    df = pd.read_csv(filename)
    df_ball = df[['system_time','frame','gameclock','period' ,'ball_x', 'ball_y', 'ball_z']]
    df_ball['player'] = 'ball'
    df_ball['id'] = 0
    df_ball['team'] = -1
    df_ball['num'] = None
    df_ball.columns = ['system_time','frame','gameclock','period','x','y','z','player','id','team',  'num']

    return df_ball

def get_home_data(filename):
    df = pd.read_csv(filename)
    # Define the columns you want to keep in the new DataFrame
    columns_to_keep = ['system_time', 'frame', 'gameclock', 'period']

    # Initialize an empty DataFrame to store the filtered data
    df_home = pd.DataFrame()
    num_players = 22
    # Iterate over player columns and add them to the list of columns to keep
    for i in range(1, num_players):  # Replace num_players with the actual number of players
        player_columns = [
            f'player{i}_team',
            f'player{i}_id',
            f'player{i}_shirt_num',
            f'player{i}_x',
            f'player{i}_y'
        ]
        # Check if player team is 0
        mask = df[f'player{i}_team'] == 0
        # If player team is 0, add corresponding player columns to home_df
        if mask.any():
            player_data = df.loc[mask, player_columns]
            df_home = pd.concat([df_home, player_data], axis=1)

    # Concatenate non-player columns with home_df
    df_home = pd.concat([df[columns_to_keep],df_home], axis=1)
    return df_home

def get_away_data(df):
    # Define the columns you want to keep in the new DataFrame
    columns_to_keep = ['system_time', 'frame', 'gameclock', 'period']

    # Initialize an empty DataFrame to store the filtered data
    df_away = pd.DataFrame()
    num_players = 22
    # Iterate over player columns and add them to the list of columns to keep
    for i in range(1, num_players):  # Replace num_players with the actual number of players
        player_columns = [
            f'player{i}_team',
            f'player{i}_id',
            f'player{i}_shirt_num',
            f'player{i}_x',
            f'player{i}_y'
        ]
        # Check if player team is 1
        mask = df[f'player{i}_team'] == 1
        # If player team is 0, add corresponding player columns to home_df
        if mask.any():
            player_data = df.loc[mask, player_columns]
            df_away = pd.concat([df_away, player_data], axis=1)



    # Concatenate non-player columns with home_df
    df_away = pd.concat([df[columns_to_keep], df_away], axis=1)
    df_away = df_away.drop(['player10_team','player10_id',	'player10_shirt_num','player10_x','player10_y'],axis=1)

    return df_away

def get_goalkeeper_data(df):
    first_columns = df[['system_time', 'frame', 'gameclock', 'period']]

    # Get the last 10 columns
    last_10_columns = df.iloc[:, -10:]

    # Combine the selected columns into a new DataFrame
    df_goal = pd.concat([first_columns, last_10_columns], axis=1)

    # Define the mapping dictionary for renaming
    mapping = {'player21': 'goal1', 'player22': 'goal2'}

    # Rename the columns using a loop
    for old_name in df_goal.columns:
        for key, value in mapping.items():
            if key in old_name:
                new_name = old_name.replace(key, value)
                df_goal.rename(columns={old_name: new_name}, inplace=True)
    
    # Separate the DataFrame into two separate DataFrames for each goal
    df_goal1 = df_goal[['system_time','frame','gameclock','period','goal1_team', 'goal1_id', 'goal1_shirt_num', 'goal1_x', 'goal1_y']]
    df_goal2 = df_goal[['system_time','frame','gameclock','period','goal2_team', 'goal2_id', 'goal2_shirt_num', 'goal2_x', 'goal2_y']]
    #df_ball = df_ball[['system_time','frame','gameclock','period','ball_x', 'ball_y']]
    # Rename the columns to have the same names for both DataFrames
    df_goal1.columns = df_goal2.columns = ['system_time','frame','gameclock','period','team', 'id', 'num', 'x', 'y']

    # Concatenate the two DataFrames
    df_concatenated = pd.concat([df_goal1, df_goal2], ignore_index=True)

    # Add a new column named "player" containing the value "goal"
    df_concatenated['player'] = 'goal'
    df_concatenated['team'] = df_concatenated['team'].replace({3: 0, 4: 1})


    # Explode the DataFrame to expand the lists in each column
    df_exploded = df_concatenated.explode('team')


    # Reset the index
    df_exploded.reset_index(drop=True, inplace=True)

    return df_exploded

def set_col_names(df):
    """ Renames the columns to have x and y suffixes."""
    cols = list(np.repeat(df.columns[3::2], 2))
    cols = [col+'_x' if i % 2 == 0 else col+'_y' for i, col in enumerate(cols)]
    cols = np.concatenate([df.columns[:3], cols])
    df.columns = cols


def to_long_form(df):
    """ Pivots a dataframe from wide-form (each player as a separate column) to long form (rows)"""
    # Melt the DataFrame to convert from wide to long form
    df = pd.melt(df, id_vars=df.columns[:4], value_vars=df.columns[4:], var_name='player')
    
    # Extract coordinate information from the 'player' column
    df['coordinate'] = df['player'].str.split('_').str[-1]
    
    # Drop rows with NaN values
    df = df.dropna(axis=0, how='any')
    
    # Extract player number from the 'player' column
    df['player'] = df['player'].str.split('_').str[0]
    
    # Pivot the DataFrame to rearrange it into long form
    df = df.pivot_table(index=['system_time', 'frame', 'gameclock', 'period', 'player'],
                        columns='coordinate', values='value').reset_index()
    
    return df

# get all players data including goalkeepers
def get_players_tracking(filename):
    df_away = get_away_data(filename)
    df_home = get_home_data(filename)
    df_goal = get_goalkeeper_data(filename)

    df_home= to_long_form(df_home)
    df_away = to_long_form(df_away)
    df_player = pd.concat([df_home, df_away], ignore_index=True)
    df_players = pd.concat([df_player, df_goal], ignore_index=True)
    
    return df_players

def get_all_data(filename):
    data_players = get_players_tracking(filename)
    data_ball = get_ball_data(filename)

    all_data = pd.concat([data_players, data_ball], ignore_index=True)

    return all_data


def get_players_info(filename):
    column_names = ['num','team','last_name','first_name','position','birthdate','in_time','out_time','event_id','id','matchEvent_id','matchTrack_id']
    player_data = pd.read_csv('data/HAC/SOCCER_PLAYERS_OPT_2568921.csv',names=column_names)
    player_data = player_data[['num','team','last_name','first_name','position','birthdate','in_time','out_time','id']]

    return player_data



def charger_donnees(nom_fichier):
    df = pd.read_csv(nom_fichier)
    return df


def prepare_data(tracking, events):
    """Prepare and clean data by converting timestamps to datetime format, and sorting by datetime."""
    # Convert system_time to datetime
    tracking['datetime'] = pd.to_datetime(tracking['system_time'], unit='ms')
    
    # Set the match date; ensure the date is correct for each match
    match_date = pd.Timestamp("2023-08-13")
    events['timestamp'] = pd.to_datetime(events['timestamp'], format='%H:%M:%S.%f')
    events['datetime'] = match_date + (events['timestamp'] - pd.Timestamp("1900-01-01"))
    
    # Reorder columns to make 'datetime' the first column
    tracking = tracking[['datetime'] + [col for col in tracking.columns if col != 'datetime']]
    events = events[['datetime'] + [col for col in events.columns if col != 'datetime']]
    
    # Sort data by datetime
    tracking = tracking.sort_values(by='datetime')
    events = events.sort_values(by='datetime')
    
    return tracking, events

# Example usage
# tracking

    
    return tracking, events
def calculer_possession_havre(df, rpz=0.3):
    """
    Ici on calcul les distances entre chaque joueur et le ballon et la possession est attribué au joueur 
    le plus proche de la du ballon si cette distance est inférieure notre rpz
    """
    
    player_columns = [f'player{i}' for i in range(1, 23)]
    distances = pd.DataFrame(index=df.index)
    for joueur in player_columns:
        joueur_x = df[f'{joueur}_x']
        joueur_y = df[f'{joueur}_y']
        ballon_x = df['ball_x']
        ballon_y = df['ball_y']
        distances[joueur] = np.sqrt((joueur_x - ballon_x)**2 + (joueur_y - ballon_y)**2)
    df['possession'] = distances.apply(lambda row: row.idxmin() if row.min() <= rpz else None, axis=1)
    df['team_possession'] = df['possession'].apply(lambda x: df.at[0, f"{x}_team"] if pd.notna(x) else None)
    #print(distances.head())
    return df

def detecter_changements_possession(df):
    """
    Les changements de possession sont détectés et marqués chaque fois que le joueur en possession est differernt du joueur précédemment en possession
    """
    df['changement_possession'] = df['possession'].ne(df['possession'].shift()).astype(int)
    changements = df[df['changement_possession'] == 1]
    #print(changements)
    return changements

def resumer_possession(df, changements):
    resume = []
    debut_trame = 0
    temps_par_frame = 0.1 

    for index, ligne in changements.iterrows():
        duree_frames = index - debut_trame
        duree_secondes = duree_frames * temps_par_frame
        resume.append({
            'debut_trame': debut_trame,
            'fin_trame': index,
            'duree_frames': duree_frames,
            'duree_secondes': duree_secondes,
            'joueur': df.loc[debut_trame, 'possession'],
            'equipe': df.loc[debut_trame, 'team_possession']
        })
        debut_trame = index

    return pd.DataFrame(resume)

def home_format_metrica(df):
    colonnes_requises = ['frame', 'period']

    for i in range(1, 11):
        colonnes_requises.append(f'player{i}_shirt_num')  
        colonnes_requises.append(f'player{i}_x')
        colonnes_requises.append(f'player{i}_y')

    df_home = df[colonnes_requises]

    return df_home

def away_format_metrica(df):
    colonnes_requises = ['frame', 'period']

    for i in range(11, 23):  
        colonnes_requises.append(f'player{i}_shirt_num')
        colonnes_requises.append(f'player{i}_x')
        colonnes_requises.append(f'player{i}_y')

    df_away = df[colonnes_requises]

    return df_away


def determine_sides(df, period):
    # Sélectionner les premières instances de chaque période pour chaque équipe
    initial_positions = df[df['period'] == period].groupby('period').head(1)
    
    # Calculer la moyenne des positions en x pour chaque équipe pour la première frame
    average_x = initial_positions.filter(regex='player\d+_x').mean(axis=1)
    
    # Déterminer le côté en fonction de la position moyenne sur le terrain
    side = 'gauche' if average_x.iloc[0] < 60 else 'droite'
    return side

def decoupage_des_coordonnes(location):
    if pd.isna(location):
        return None, None
    location = location.strip('[]') 
    parts = location.split() 
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    else:
        return None, None

def calculer_possession_metrica(df, rpz=0.1):
    home_players = [f'Home_{i}' for i in range(1, 15)]
    away_players = [f'Away_{i}' for i in range(15, 29)]
    player_columns = home_players + away_players


    distances = pd.DataFrame(index=df.index)
    ballon_x = df['ball_x'].fillna(method='ffill')
    ballon_y = df['ball_y'].fillna(method='ffill')

    for joueur in player_columns:
        joueur_x = df[f'{joueur}_x'].fillna(method='ffill')
        joueur_y = df[f'{joueur}_y'].fillna(method='ffill')
        distances[joueur] = np.sqrt((joueur_x - ballon_x)**2 + (joueur_y - ballon_y)**2)
        
    df['possession'] = distances.idxmin(axis=1).where(distances.min(axis=1) <= rpz)
    df['team_possession'] = df['possession'].map(lambda x: 'Home' if 'Home' in str(x) else 'Away' if 'Away' in str(x) else None)

    return df

def passes_player(df, team_name, period, player_name):
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

def passe_entre_joueurs(df, team_name, period, player_name=None, recipient_name=None):
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

   
    if player_name and recipient_name:
        data = df[(df['team.name'] == team_name) & (df['period'] == period) &
                (df['player.name'] == player_name) & (df['pass.recipient.name'] == recipient_name) &
                df['location.x'].notna() & df['location.y'].notna() &
                df['pass.end_location_x'].notna() & df['pass.end_location_y'].notna()]
    else:
        data = df[(df['team.name'] == team_name) & (df['period'] == period) &
                df['location.x'].notna() & df['location.y'].notna() &
                df['pass.end_location_x'].notna() & df['pass.end_location_y'].notna()]

    for _, pass_ in data.iterrows():
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
        title=f"Les passes entre {player_name} et {recipient_name} durant la {period} période de jeu",
        width=1000,
        height=700
    )

    fig.show()


def affiche_tirs(df, home_team, away_team, selected_period):
    fig = go.Figure()
    terrain_shapes = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="Black")),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="Black")),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="Black")),
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


