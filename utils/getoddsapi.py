import requests
import csv
import pandas as pd
from datetime import datetime
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Your API key
API_KEY = os.getenv('API_KEY')

# Team abbreviation mapping (you can add more as needed)
team_abbreviations = {
    'Detroit Pistons': 'DET',
    'Indiana Pacers': 'IND',
    'Toronto Raptors': 'TOR',
    'Cleveland Cavaliers': 'CLE',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Chicago Bulls': 'CHI',
    'Charlotte Hornets': 'CHA',
    'New York Knicks': 'NYK',
    'Orlando Magic': 'ORL',
    'Philadelphia Sixers': 'PHI',
    'Brooklyn Nets': 'BKN',
    'Boston Celtics': 'BOS',
    'Atlanta Hawks': 'ATL',
    'Washington Wizards': 'WAS',
    'Denver Nuggets': 'DEN',
    'Utah Jazz': 'UTA',
    'Oklahoma City Thunder': 'OKC',
    'Portland Trail Blazers': 'POR',
    'Minnesota Timberwolves': 'MIN',
    'Los Angeles Lakers': 'LAL',
    'Los Angeles Clippers': 'LAC',
    'Sacramento Kings': 'SAC',
    'Phoenix Suns': 'PHX',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Memphis Grizzlies': 'MEM',
    'Dallas Mavericks': 'DAL',
    'San Antonio Spurs': 'SAS',
    'New Orleans Pelicans': 'NOP'
}

seen = set()

player_teams = pd.read_csv('utils/nba_players_teams.csv')


# Function to get a list of NBA events
def get_nba_events():
    url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
    
    # Parameters to fetch the NBA events
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'oddsFormat': 'american',
        'markets': 'h2h',  # Only fetching event info (head-to-head)
        'dateFormat': 'iso'
    }

    # Making the request to fetch events
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        events_data = response.json()
        # Return list of event IDs and team information
        events = [(event['id'], event['home_team'], event['away_team']) for event in events_data]
        return events
    else:
        print(f"Error fetching events: {response.status_code}")
        return []

# Function to get player props (rebounds) for a specific event, filtering only FanDuel
def get_player_props(event_id, home_team, away_team, csv_writer):
    url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds'
    
    # Parameters to get player prop odds (rebounds)
    params = {
        'apiKey': API_KEY,
        'markets': 'player_rebounds',  # Market for player rebounds
        'oddsFormat': 'american',
        'regions': 'us'
    }

    # Making the request to fetch player props
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        player_props = response.json()

        # Filter for FanDuel as the bookmaker
        fanduel_data = next((bookmaker for bookmaker in player_props.get('bookmakers', []) if bookmaker['title'] == 'FanDuel'), None)
        
        if fanduel_data:
            print(f"Fetching FanDuel odds for {home_team} vs {away_team}")
            for market in fanduel_data['markets']:
                if market['key'] == 'player_rebounds':
                    for outcome in market['outcomes']:
                        player_name = outcome['description']
                        betting_line = outcome['point']
                        odds = outcome['price']
                        print(f"Player: {player_name}")
                        if home_team == 'Philadelphia 76ers':
                            home_team = 'Philadelphia Sixers'
                        if away_team == 'Philadelphia 76ers':
                            away_team = 'Philadelphia Sixers'
                        if player_name.split(' ')[-1] == 'Jr': 
                            player_name += '.'
                        if player_teams[player_teams['Player'] == player_name].empty:
                            print(f"Player {player_name} not found in the player_teams CSV file.")
                            continue
                        player_team = player_teams[player_teams['Player'] == player_name]['Team'].values[0]
                        player_team_abbr = team_abbreviations[player_team]
                        opp_team = away_team if player_team == home_team else home_team
                        opp_team_abbr = team_abbreviations[opp_team]

                        # Check if the player is playing at home and get abbreviations for teams, home if listed team is first
                        is_home = 1 if player_team == home_team else 0
                        
                        
                        # If player name in the csv file already, just update other odds
                        # Otherwise, write a new row
                        if player_name in seen:
                            data_so_far[4 if outcome['name'] == 'Over' else 5] = odds
                            csv_writer.writerow(data_so_far)
                        else:
                            data_so_far = [player_name, player_team_abbr, opp_team_abbr, betting_line, odds if outcome['name'] == 'Over' else '', odds if outcome['name'] == 'Under' else '', 1 if is_home else 0]
                            seen.add(player_name)
        else:
            # If no FanDuel odds, terminate program
            print(f"No FanDuel odds available for {home_team} vs {away_team}. Terminating program.")
            exit(0)
    else:
        # Print the error code and inspect the response
        print(f"Error fetching player props for {home_team} vs {away_team} (Event ID: {event_id}) - Status code: {response.status_code}")
        print(f"Response content: {response.text}")

# Main function to loop through each event and get player props
def get_props_for_all_events():
    current_date = datetime.today().strftime('%Y-%m-%d')
    # Create the CSV file and write the header
    with open(f'nba_player_rebounds_odds-{current_date}.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Player Name', 'Player Team', 'Opposing Team', 'Rebounds Line', 'Over Odds', 'Under Odds', 'Home/Away'])

        # Fetch NBA events
        events = get_nba_events()

        # Loop through each event and fetch player props
        for event_id, home_team, away_team in events:
            print(f"\nFetching player props for {home_team} vs {away_team} (Event ID: {event_id})")
            get_player_props(event_id, home_team, away_team, csv_writer)

# Run the main function
get_props_for_all_events()
