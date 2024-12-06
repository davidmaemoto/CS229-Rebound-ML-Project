import requests
from bs4 import BeautifulSoup
import csv

# Define the URL to scrape
url = 'https://basketball.realgm.com/nba/players'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the page content
    soup = BeautifulSoup(response.content, 'html.parser')
    # Locate the table
    table = soup
    
    # Open a CSV file to write the results
    with open('nba_players_teams.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Player", "Team"])  # Write the header
        
        # Loop through each row in the table
        for row in table.find_all('tr'):
            # Extract player name
            player_element = row.select_one("td[data-th='Player'] a")
            player_name = player_element.text if player_element else "N/A"
            
            # Extract team name
            team_element = row.select_one("td[data-th='Current Team'] a")
            team_name = team_element.text if team_element else "N/A"
            
            # Write to the CSV
            writer.writerow([player_name, team_name])
else:
    print(f"Failed to retrieve data: {response.status_code}")
