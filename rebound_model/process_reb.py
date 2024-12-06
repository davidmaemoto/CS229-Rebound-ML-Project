import pandas as pd
import numpy as np
import json

def preprocess_data():
    # Separate data into training and testing sets
    seasons = [2018, 2019, 2020, 2021, 2024]
    # convert the data to a pandas dataframe
    games_details = pd.read_csv('data/games_details.csv')
    games = pd.read_csv('data/games.csv')
    player_shooting = pd.read_csv('data/player_shooting.csv')
    opp_stats = pd.read_csv('data/opp_stats.csv')
    player_per_game = pd.read_csv('data/player_per_game.csv')
    team_abbrev = pd.read_csv('data/team_abbrev.csv')
    team_summary = pd.read_csv('data/team_summary.csv')

    # Get the game details for the 2018, 2019, 2020, and 2021 seasons
    filtered_games = games[games['SEASON'].between(2018, 2021)]

    # Create the desired dictionary
    season_game_map = {}

    # Loop through the filtered DataFrame
    for season, group in filtered_games.groupby('SEASON'):
        season_game_map[season] = {}
        for _, row in group.iterrows():
            game_id = row['GAME_ID']
            home_team_id = row['HOME_TEAM_ID']
            visitor_team_id = row['VISITOR_TEAM_ID']
            # Handle duplicates by ensuring each game_id is unique
            if game_id not in season_game_map[season]:
                season_game_map[season][game_id] = {
                    'HOME_TEAM_ID': home_team_id,
                    'VISITOR_TEAM_ID': visitor_team_id
                }


    # save to json file
    with open('data/season_game_map.json', 'w') as f:
        json.dump(season_game_map, f)



    # Step 1: Filter for players who meet x3pa_per_game >= 4.0 for all seasons (2018-2021)
    qualified_players = None

    for season in seasons:
        filtered_season_players = player_per_game[
            (player_per_game['season'] == season) &
            (player_per_game['mp_per_game'] >= 15.0)
        ]
        
        # Get the set of player names for this season
        season_player_names = set(filtered_season_players['player'])
        
        # Perform intersection for each season to ensure players meet the condition in all seasons
        if qualified_players is None:
            qualified_players = season_player_names
        else:
            qualified_players = qualified_players.intersection(season_player_names)

    # Step 2: Filter players for the 2024 season
    players_2024 = player_per_game[player_per_game['season'] == 2024]
    player_names_2024 = set(players_2024['player'])

    # Step 3: Get the intersection with players from 2024
    shooters = list(qualified_players.intersection(player_names_2024))
    print(len(shooters))
    # Save the list of player names to json
    with open('data/shooters.json', 'w') as f:
        json.dump(shooters, f)

    # For each season 2018 - 2021, build a feature vector for each player
    feature_dict ={season: {} for season in seasons}

    # Add shooting data features first
    for season in seasons:
        # Filter the player_shooting data for the current season
        season_data = player_shooting[player_shooting['season'] == season]
        
        # Filter the players from the intersection (shooters)
        for player in shooters:
            # Filter for this specific player in the current season
            player_data = season_data[season_data['player'] == player]
            
            if not player_data.empty:
                # Extract the relevant statistics
                avg_dist_fga = player_data['avg_dist_fga'].values[0]
                percent_fga_from_x0_3_range = player_data['percent_fga_from_x0_3_range'].values[0]
                fg_percent_from_x0_3_range = player_data['fg_percent_from_x0_3_range'].values[0]
                fg_percent = player_data['fg_percent'].values[0]
                experience = float(player_data['experience'].values[0])
                
                # Construct the feature vector and add it to the dictionary for the season
                feature_dict[season][player] = {
                    'avg_dist_fga': avg_dist_fga if not np.isnan(avg_dist_fga) else 0.0,
                    'percent_fga_from_x0_3_range': percent_fga_from_x0_3_range if not np.isnan(percent_fga_from_x0_3_range) else 0.0,
                    'fg_percent_from_x0_3_range': fg_percent_from_x0_3_range if not np.isnan(fg_percent_from_x0_3_range) else 0.0,
                    'fg_percent': fg_percent if not np.isnan(fg_percent) else 0.0,
                    'experience': experience if not np.isnan(experience) else 0.0
                }
        

    # Add player per game features
    for season in seasons:
        # Filter the player_per_game data for the current season
        season_per_game_data = player_per_game[player_per_game['season'] == season]
        
        # Loop through each player in the filtered list for that season
        for player in shooters:
            # Filter for this specific player in the current season
            player_per_game_data = season_per_game_data[season_per_game_data['player'] == player]
            
            if not player_per_game_data.empty:
                # Extract the additional statistics
                usage_game = player_per_game_data['mp_per_game'].values[0]
                trb_per_game = player_per_game_data['trb_per_game'].values[0]
                blk_per_game = player_per_game_data['blk_per_game'].values[0]
                fta_per_game = player_per_game_data['fta_per_game'].values[0]
                pos = player_per_game_data['pos'].values[0]
                pos_mp = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
                if '-' in pos:
                    pos = pos.split('-')
                    pos = [pos_mp[p] for p in pos]
                    pos = np.mean(pos)
                else:
                    pos = pos_mp[pos]
                
                # Update the existing feature vector dictionary with the new stats
                if player in feature_dict[season]:
                    feature_dict[season][player].update({
                        'Usage_game': usage_game if not np.isnan(usage_game) else 0.0,
                        'Trb_per_game': trb_per_game if not np.isnan(trb_per_game) else 0.0,
                        'Blk_per_game': blk_per_game if not np.isnan(blk_per_game) else 0.0,
                        'Fta_per_game': fta_per_game if not np.isnan(fta_per_game) else 0.0,
                        'Pos': pos if not np.isnan(pos) else 0.0
                    })

    # Save the feature dictionary to a json file
    with open('data/player_feature_dict.json', 'w') as f:
        json.dump(feature_dict, f)


    # Team features
    # Step 1: Create a dictionary to hold the results
    team_ratios = {}

    # Step 3: Calculate team ratios for each season
    for season in seasons:
        # Create a new dictionary for the season
        team_ratios[season] = {}

        # Filter teams for the current season
        teams = team_summary[team_summary['season'] == season]

        for index, row in teams.iterrows():
            if row['team'] != 'League Average':  # Skip league average row
                team_abbr = row['abbreviation']
                offensive_rating_ratio = row['o_rtg'] 
                defensive_rating_ratio = row['d_rtg']
                pace_ratio = row['pace']
                x3par = row['x3p_ar']
                oreb_pct = row['orb_percent']
                
                # Fetch opponent stats for the corresponding team
                opp_stats_row = opp_stats[(opp_stats['season'] == season) & (opp_stats['abbreviation'] == team_abbr)]
                opp_x3pa_per_game = opp_stats_row['opp_x3pa_per_game'].values[0] if not opp_stats_row.empty else None
                opp_orb_per_game = opp_stats_row['opp_orb_per_game'].values[0] if not opp_stats_row.empty else None
                opp_drb_per_game = opp_stats_row['opp_drb_per_game'].values[0] if not opp_stats_row.empty else None
                if team_abbr == "CHO":
                    team_abbr = "CHA"
                if team_abbr == "BRK":
                    team_abbr = "BKN"
                if team_abbr == "PHO":
                    team_abbr = "PHX"
                # Store the ratios and the Opp x3pa_per_game in the inner dictionary
                team_ratios[season][team_abbr] = {
                    'Offensive_Rating_Ratio': offensive_rating_ratio,
                    'Pace_Ratio': pace_ratio,
                    'Opp_x3pa_per_game': opp_x3pa_per_game,
                    'Defensive_Rating_Ratio': defensive_rating_ratio,
                    'X3par': x3par,
                    'Oreb_pct': oreb_pct,
                    'Opp_orb_per_game': opp_orb_per_game,
                    'Opp_drb_per_game': opp_drb_per_game
                }


    # Save team features json
    with open('data/team_features.json', 'w') as f:
        json.dump(team_ratios, f)


    # filter the relevant games only
    filtered_game_details = pd.DataFrame()
    for season in range(2018, 2022):
        # Filter rows where PLAYER_NAME is in shooters_set
        potential_game_ids = season_game_map[season].keys()
        season_games_details = games_details[
            games_details['PLAYER_NAME'].isin(shooters) &
            games_details['GAME_ID'].isin(potential_game_ids) &
            games_details['COMMENT'].isnull()
        ]
        # add a SEASON column
        season_games_details['SEASON'] = season
        filtered_game_details = pd.concat([filtered_game_details, season_games_details])

    pd.DataFrame(filtered_game_details).to_csv('data/all_data.csv', index=False)
    



def get_player_features(player_name, team, opp, home):
    games_details = pd.read_csv('data/games_details.csv')
    games = pd.read_csv('data/games.csv')
    player_shooting = pd.read_csv('data/player_shooting.csv')
    opp_stats = pd.read_csv('data/opp_stats.csv')
    player_per_game = pd.read_csv('data/player_per_game.csv')
    team_abbrev = pd.read_csv('data/team_abbrev.csv')
    team_summary = pd.read_csv('data/team_summary.csv')
    feature_vec = []
    season = 2024
    season_data = player_shooting[player_shooting['season'] == season]
    player_data = season_data[season_data['player'] == player_name]
    if not player_data.empty:
        avg_dist_fga = player_data['avg_dist_fga'].values[0]
        percent_fga_from_x0_3_range = player_data['percent_fga_from_x0_3_range'].values[0]
        fg_percent_from_x0_3_range = player_data['fg_percent_from_x0_3_range'].values[0]
        fg_percent = player_data['fg_percent'].values[0]
        experience = float(player_data['experience'].values[0])
        feature_vec.extend([avg_dist_fga, percent_fga_from_x0_3_range, fg_percent_from_x0_3_range, fg_percent, experience])
    else:
        return None
    season_per_game_data = player_per_game[player_per_game['season'] == season]
    player_per_game_data = season_per_game_data[season_per_game_data['player'] == player_name]
    if not player_per_game_data.empty:
        usage_game = player_per_game_data['mp_per_game'].values[0]
        trb_per_game = player_per_game_data['trb_per_game'].values[0]
        blk_per_game = player_per_game_data['blk_per_game'].values[0]
        fta_per_game = player_per_game_data['fta_per_game'].values[0]
        pos = player_per_game_data['pos'].values[0]
        pos_mp = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
        if '-' in pos:
            pos = pos.split('-')
            pos = [pos_mp[p] for p in pos]
            pos = np.mean(pos)
        else:
            pos = pos_mp[pos]
        feature_vec.extend([usage_game, trb_per_game, blk_per_game, fta_per_game, pos])
    else:
        feature_vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    teams = team_summary[team_summary['season'] == season]
    if team == "CHA":
        team = "CHO"
    if team == "BKN":
        team = "BRK"
    if team == "PHX":
        team = "PHO"
    if opp == "CHA":
        opp = "CHO"
    if opp == "BKN":
        opp = "BRK"
    if opp == "PHX":
        opp = "PHO"
    team_row = teams[teams['abbreviation'] == team]
    if not team_row.empty:
        offensive_rating_ratio = team_row['o_rtg'].values[0]
        defensive_rating_ratio = team_row['d_rtg'].values[0]
        pace_ratio = team_row['pace'].values[0]
        x3par = team_row['x3p_ar'].values[0]
        oreb_pct = team_row['orb_percent'].values[0]
        opp_stats_row = opp_stats[(opp_stats['season'] == season) & (opp_stats['abbreviation'] == team)]
        team_stats_row = opp_stats[(opp_stats['season'] == season) & (opp_stats['abbreviation'] == opp)]
        opp_x3pa_per_game = opp_stats_row['opp_x3pa_per_game'].values[0] if not opp_stats_row.empty else 0.0
        opp_orb_per_game = team_stats_row['opp_orb_per_game'].values[0] if not team_stats_row.empty else 0.0
        opp_drb_per_game = team_stats_row['opp_drb_per_game'].values[0] if not team_stats_row.empty else 0.0
        feature_vec.extend([offensive_rating_ratio, pace_ratio, opp_x3pa_per_game, defensive_rating_ratio, x3par, oreb_pct, opp_orb_per_game, opp_drb_per_game])
    else:
        feature_vec.extend([0.0 for _ in range(8)])
    if home:
        feature_vec.append(1.0)
    else:
        feature_vec.append(0.0)
    return feature_vec

