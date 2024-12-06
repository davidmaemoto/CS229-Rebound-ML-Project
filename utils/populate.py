import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page to scrape
url = "https://basketballmonster.com/Boxscores.aspx"

# Send a GET request to fetch the raw HTML content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# List to store player data from all tables
all_player_data = []

# Find all tables with the specified class
tables = soup.find_all('table', {'class': 'table-bordered table-hover table-sm base-td-small datatable ml-0'})

# Iterate through each table and extract the player data from the tbody
for table in tables[1:]:
    # Find all rows within the tbody of each table
    rows = table.find_all('tr')
    for row in rows[1:]:
        # Find the player name and the corresponding rebound number
        player_name_tag = row.find('a')  # Assuming player names are inside anchor <a> tags
        if player_name_tag:
            player_name = player_name_tag.text
            if player_name == 'O.G. Anunoby':
                player_name = 'OG Anunoby'
            if player_name == 'Cam Johnson':
                player_name = 'Cameron Johnson'
            if player_name == 'Kelly Oubre':
                player_name = 'Kelly Oubre Jr.'
            if player_name == 'Jabari Smith Jr':
                player_name = 'Jabari Smith Jr.'
            if player_name == 'PJ Washington':
                player_name = 'P.J. Washington'
            print(player_name)
            rebound_td = row.find_all('td')[12]  # Adjust the index based on the column of rebounds
            rebounds = rebound_td.text.strip()
            print(rebounds)
            all_player_data.append({'Player': player_name, 'Rebounds': rebounds})

# Convert the collected data to a DataFrame
df = pd.DataFrame(all_player_data)


#### THIS MUST BE CHANGED ####
to_populate = pd.read_csv('predictions_data/predictions_rebounding-2024-12-03.csv')
for row in to_populate.iterrows():
    player = row[1]['Player']
    if player in df['Player'].values:
        rebounds = df[df['Player'] == player]['Rebounds'].values[0]
        to_populate.loc[to_populate['Player'] == player, 'Rebounds'] = rebounds
# Save the updated DataFrame to a new CSV file
#### THIS MUST BE CHANGED ####
to_populate.to_csv('predictions_data/predictions_rebounding-2024-12-03.csv', index=False)
