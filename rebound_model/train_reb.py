import pandas as pd
import json
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def train_model():
    with open('data/shooters.json', 'r') as f:
        shooters = json.load(f)
    with open ('data/season_game_map.json', 'r') as f:
        season_game_map = json.load(f)
    with open('data/team_features.json', 'r') as f:
        team_features = json.load(f)
    with open('data/player_feature_dict.json', 'r') as f:
        player_feature_dict = json.load(f)  
    games_details = pd.read_csv('data/filtered_games_details.csv')
    unfiltered_game_details = pd.read_csv('data/games_details.csv')
    seasons = [2018, 2019, 2020, 2021]

    X = []
    y = []

    for season in seasons:
        for game_id in set(games_details[games_details['SEASON'] == season]['GAME_ID']):
            home_team_id = season_game_map[str(season)][str(game_id)]['HOME_TEAM_ID']
            away_team_id = season_game_map[str(season)][str(game_id)]['VISITOR_TEAM_ID']
            home_team_abbrev = unfiltered_game_details[unfiltered_game_details['TEAM_ID'] == home_team_id]['TEAM_ABBREVIATION'].values[0]
            away_team_abbrev = unfiltered_game_details[unfiltered_game_details['TEAM_ID'] == away_team_id]['TEAM_ABBREVIATION'].values[0]
            for shooter in shooters:
                for i in range(2):
                    if i == 1:
                        team_abbrev = home_team_abbrev
                        team_id = home_team_id
                        opp_team_abbrev = away_team_abbrev
                        opp_team_id = away_team_id
                    else:
                        team_abbrev = away_team_abbrev
                        team_id = away_team_id
                        opp_team_abbrev = home_team_abbrev
                        opp_team_id = home_team_id
                    # ensure that min column exists and is greater than 0
                    # do not inlcude if there is something in the COMMENT column
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
                        X.append(feature_vec)
                        y.append(reb)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    regression_model = HuberRegressor(epsilon=3).fit(X, y)
    xg_boost_model = XGBRegressor().fit(X, y)
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.5).fit(X, y)
    # Other models I experimented with:
    #ransac_model = RANSACRegressor().fit(X, y)
    #random_forest_model = RandomForestRegressor(n_estimators=100).fit(X, y)
    #mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000).fit(X, y)
    return regression_model, xg_boost_model, svr_model, "" , "", "", scaler, X, y