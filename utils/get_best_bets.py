import pandas as pd
from datetime import datetime
import sys

def get_best_bets(file_path):
    """
    Read a CSV file and return rows with 'Edge Over' > 0.2 or 'Edge Under' > 0.2.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the filtered rows.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Go through each row and get max of edge over and edge under, utilize this as some sort of unregularized probability of being chosen such that bets with higher edge have higher probability of being chosen, must be > 0
    df['Max Edge'] = df[['Edge Over', 'Edge Under']].max(axis=1)
    # only consider rows where max edge is > 0.05
    df = df[df['Max Edge'] > 0.05]
    # Sort by max edge
    df = df.sort_values('Max Edge', ascending=False)
    # Draw from distribution of edges
    df['Edge CDF'] = df['Max Edge'].cumsum()
    df['Edge CDF'] = df['Edge CDF'] / df['Edge CDF'].max()
    df['Edge CDF'] = df['Edge CDF'].shift(1)
    df['Edge CDF'] = df['Edge CDF'].fillna(0)
    df['Edge CDF'] = df['Edge CDF'] * 100
    df['Edge CDF'] = df['Edge CDF'].astype(int)
    # Draw 10 bets, return these rows
    df = df.sample(n=min(10, len(df)-1), weights='Edge CDF')
    # Sort by max edge
    df = df.sort_values('Max Edge', ascending=False)
    # Add a column for how much you would win if you bet 1 dollar, rounded up to nearest cent based on implied probability (this is a string that has a percent sign at the end)
    
    # convert implied probability to betting odds, then calculate how much you would win if you bet 1 dollar, rounded up to nearest cent
    df['Implied Probability Over'] = df['Implied Probability Over (from odds)'].str.rstrip('%').astype('float') / 100
    df['Implied Probability Under'] = df['Implied Probability Under (from odds)'].str.rstrip('%').astype('float') / 100
    df['Bet Over'] = round(1 / df['Implied Probability Over'], 2) -1 
    df['Bet Under'] = round(1 / df['Implied Probability Under'], 2) -1

    




    #filtered_df = df[((df['Edge Over'] > 0.175) & (df['Edge Over'] < 0.225)) | ((df['Edge Under'] > 0.175) & (df['Edge Under'] < 0.225))]
    #filtered_df = df[((df['Edge Over'] > 0.075) & (df['Edge Over'] < 0.125)) | ((df['Edge Under'] > 0.075) & (df['Edge Under'] < 0.125))]
    #filtered_df = df[((df['Edge Over'] > 0.18) & (df['Edge Over'] < 0.25)) | ((df['Edge Under'] > 0.18) & (df['Edge Under'] < 0.25))]
    #filtered_df = df[((df['Edge Over'] > 0.15) & (df['Edge Over'] < 0.20)) | ((df['Edge Under'] > 0.15) & (df['Edge Under'] < 0.20))]
    return df

if __name__ == "__main__":
    current_date = datetime.today().strftime('%Y-%m-%d')
    result_df = get_best_bets(sys.argv[1])
    result_df.to_csv(f'best_bets-{current_date}.csv', index=False)
