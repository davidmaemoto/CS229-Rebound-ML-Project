import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

def test(scaler):
    with open('data/shooters.json', 'r') as f:
        shooters = json.load(f)
    with open ('data/season_game_map.json', 'r') as f:
        season_game_map = json.load(f)
    with open('data/team_features.json', 'r') as f:
        team_features = json.load(f)
    with open('data/player_feature_dict.json', 'r') as f:
        player_feature_dict = json.load(f)  
    games_details = pd.read_csv('data/test_set.csv')
    unfiltered_game_details = pd.read_csv('data/games_details.csv')
    seasons = [2018, 2019, 2020, 2021]

    
    X_test = []
    y_test = []

    for season in seasons:
        for game_id in set(games_details[games_details['SEASON'] == season]['GAME_ID']):
            home_team_id = season_game_map[str(season)][str(game_id)]['HOME_TEAM_ID']
            away_team_id = season_game_map[str(season)][str(game_id)]['VISITOR_TEAM_ID']
            home_team_abbrev = unfiltered_game_details[unfiltered_game_details['TEAM_ID'] == home_team_id]['TEAM_ABBREVIATION'].values[0]
            away_team_abbrev = unfiltered_game_details[unfiltered_game_details['TEAM_ID'] == away_team_id]['TEAM_ABBREVIATION'].values[0]
            for shooter in shooters:
                for i in range(2):
                    if i == 0:
                        team_abbrev = home_team_abbrev
                        team_id = home_team_id
                        opp_team_abbrev = away_team_abbrev
                        opp_team_id = away_team_id
                    else:
                        team_abbrev = away_team_abbrev
                        team_id = away_team_id
                        opp_team_abbrev = home_team_abbrev
                        opp_team_id = home_team_id
                    player_data = games_details[
                        (games_details['GAME_ID'] == game_id) & 
                        (games_details['TEAM_ID'] == team_id) &
                        (games_details['PLAYER_NAME'] == shooter)
                    ]
                    if not player_data.empty:
                        reb = player_data['REB'].values[0]
                        feature_vec = []
                        for value in player_feature_dict[str(season)][shooter].values():
                            feature_vec.append(value)
                        feature_vec.append(team_features[str(season)][team_abbrev]['Offensive_Rating_Ratio'])
                        feature_vec.append(team_features[str(season)][team_abbrev]['Pace_Ratio'])
                        feature_vec.append(team_features[str(season)][team_abbrev]['Opp_x3pa_per_game'])
                        feature_vec.append(team_features[str(season)][team_abbrev]['Defensive_Rating_Ratio'])
                        feature_vec.append(team_features[str(season)][team_abbrev]['X3par'])
                        feature_vec.append(team_features[str(season)][team_abbrev]['Oreb_pct'])
                        feature_vec.append(team_features[str(season)][opp_team_abbrev]['Opp_orb_per_game'])
                        feature_vec.append(team_features[str(season)][opp_team_abbrev]['Opp_drb_per_game'])
                        feature_vec.append(float(i))
                        X_test.append(feature_vec)
                        y_test.append(reb)

    X_test = scaler.transform(X_test)
    return X_test, y_test

                        