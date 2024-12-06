import utils.evaluate as evaluate
import pandas as pd
import matplotlib.pyplot as plt
import sys
import rebound_model.train_reb as train_reb
import rebound_model.test_reb as test_reb
import rebound_model.process_reb as process_reb
import joblib
from scipy.stats import norm
import utils.implied_odds as implied_odds
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def train_test():
    error = {}
    num_folds = 5
    for iter in range(5):
        process_reb.preprocess_data()
        print("Preprocessing done")
        filtered_game_details = pd.read_csv('data/all_data.csv')
        train_data = filtered_game_details.sample(frac=0.7, random_state=42)
        test_data = filtered_game_details.drop(train_data.index)
        train_data.to_csv('data/filtered_games_details.csv', index=False)
        test_data.to_csv('data/test_set.csv', index=False)
        regression_model, xg_boost_model, svr_model,_, _,_, scaler, X_train, y_train = train_reb.train_model()
        joblib.dump(regression_model, 'regression_model.pkl')
        joblib.dump(xg_boost_model, 'xg_boost_model.pkl')
        joblib.dump(svr_model, 'svr_model.pkl')
        print("Training done")
        X_test, y_test = test_reb.test(scaler)
        joblib.dump(scaler, 'scaler.pkl')
        print("Testing done")

        y_test_reg_pred = regression_model.predict(X_test)
        y_test_xgb_pred = xg_boost_model.predict(X_test)
        y_test_svr_pred = svr_model.predict(X_test)

        y_train_reg_pred = regression_model.predict(X_train)
        y_train_xgb_pred = xg_boost_model.predict(X_train)
        y_train_svr_pred = svr_model.predict(X_train)
        
        for i in range(len(y_test)):
            y_test_reg_pred[i] = max(0, y_test_reg_pred[i])
            y_test_xgb_pred[i] = max(0, y_test_xgb_pred[i])
            y_test_svr_pred[i] = max(0, y_test_svr_pred[i])
        
        for i in range(len(y_train)):
            y_train_reg_pred[i] = max(0, y_train_reg_pred[i])
            y_train_xgb_pred[i] = max(0, y_train_xgb_pred[i])
            y_train_svr_pred[i] = max(0, y_train_svr_pred[i])

            
        mae, mse, rmse, r2 = evaluate.test_regression(y_test, y_test_reg_pred)
        # if regression test there already, else add it
        if 'regression_test' not in error:
            error['regression_test'] = [mae, mse, rmse, r2]
        else:
            error['regression_test'][0] += mae
            error['regression_test'][1] += mse
            error['regression_test'][2] += rmse
            error['regression_test'][3] += r2
        print("Results for regression model:")
        print("MAE: ", mae, "MSE: ", mse, "RMSE: ", rmse, "R2: ", r2)
        mae, mse, rmse, r2 = evaluate.test_regression(y_test, y_test_xgb_pred)


        if 'xg_boost_test' not in error:
            error['xg_boost_test'] = [mae, mse, rmse, r2]
        else:
            error['xg_boost_test'][0] += mae
            error['xg_boost_test'][1] += mse
            error['xg_boost_test'][2] += rmse
            error['xg_boost_test'][3] += r2
        print("Results for XGBoost model:")
        print("MAE: ", mae, "MSE: ", mse, "RMSE: ", rmse, "R2: ", r2)
        mae, mse, rmse, r2 = evaluate.test_regression(y_test, y_test_svr_pred)
        if 'svr_test' not in error:
            error['svr_test'] = [mae, mse, rmse, r2]
        else:
            error['svr_test'][0] += mae
            error['svr_test'][1] += mse
            error['svr_test'][2] += rmse
            error['svr_test'][3] += r2
        print("Results for SVR model:")
        print("MAE: ", mae, "MSE: ", mse, "RMSE: ", rmse, "R2: ", r2)

        mae, mse, rmse, r2 = evaluate.test_regression(y_train, y_train_reg_pred)
        if 'regression_train' not in error:
            error['regression_train'] = [mae, mse, rmse, r2]
        else:
            error['regression_train'][0] += mae
            error['regression_train'][1] += mse
            error['regression_train'][2] += rmse
            error['regression_train'][3] += r2
        print("Results for regression model (train):")
        print("MAE: ", mae, "MSE: ", mse, "RMSE: ", rmse, "R2: ", r2)
        mae, mse, rmse, r2 = evaluate.test_regression(y_train, y_train_xgb_pred)
        if 'xg_boost_train' not in error:
            error['xg_boost_train'] = [mae, mse, rmse, r2]
        else:
            error['xg_boost_train'][0] += mae
            error['xg_boost_train'][1] += mse
            error['xg_boost_train'][2] += rmse
            error['xg_boost_train'][3] += r2
        print("Results for XGBoost model (train):")
        print("MAE: ", mae, "MSE: ", mse, "RMSE: ", rmse, "R2: ", r2)
        mae, mse, rmse, r2 = evaluate.test_regression(y_train, y_train_svr_pred)
        if 'svr_train' not in error:
            error['svr_train'] = [mae, mse, rmse, r2]
        else:
            error['svr_train'][0] += mae
            error['svr_train'][1] += mse
            error['svr_train'][2] += rmse
            error['svr_train'][3] += r2
        print("Results for SVR model (train):")
        print("MAE: ", mae, "MSE: ", mse, "RMSE: ", rmse, "R2: ", r2)
        if iter == 0:
            create_heatmap(y_test, y_test_reg_pred, 'Regression Model', 'regression_heatmap.png')
            create_heatmap(y_test, y_test_xgb_pred, 'XGBoost Model', 'xgboost_heatmap.png')
            create_heatmap(y_test, y_test_svr_pred, 'SVR Model', 'svr_heatmap.png')

    for key in error:
        for i in range(len(error[key])):
            error[key][i] /= num_folds
    print("Final results:")
    print(error)
    joblib.dump(error, 'error.pkl')

import seaborn as sns
# Function to create and save heatmaps
def create_heatmap(actual, predicted, title, filename):
    plt.figure(figsize=(8, 6))
    sns.histplot(x=actual, y=predicted, bins=20, cmap='viridis', cbar=True)
    plt.plot([0, 20], [0, 20], 'k--', linewidth=2, label='Perfect Prediction Line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def predict(player_names, teams, opps, lines, odds_overs, odds_unders, homes):
    regression_model = joblib.load('regression_model.pkl')
    xg_boost_model = joblib.load('xg_boost_model.pkl')
    svr_model = joblib.load('svr_model.pkl')
    error = joblib.load('error.pkl')
    scaler = joblib.load('scaler.pkl')
    current_date = datetime.today().strftime('%Y-%m-%d')
    results = []

    for i in range(len(player_names)):
        player_name = player_names[i]
        print("Predicting for player: ", player_name)
        team = teams[i]
        opp = opps[i]
        line = lines[i]
        odds_over = float(odds_overs[i])
        odds_under = float(odds_unders[i])
        home = homes[i]
        feature_vec = process_reb.get_player_features(player_name, team, opp, home)
        if feature_vec == None:
            print("Rookie or not enough data, skipping")
            continue
        print("Feature vector (unscaled): ", feature_vec)
        feature_vec = scaler.transform([feature_vec])
        print("Feature vector: ", feature_vec)
        prediction = {
            'regression': max(0, regression_model.predict(feature_vec)),
            'xg_boost': max(0, xg_boost_model.predict(feature_vec)),
            'svr': max(0, svr_model.predict(feature_vec)),
            'CONSENSUS PICK': (regression_model.predict(feature_vec) + xg_boost_model.predict(feature_vec) + svr_model.predict(feature_vec)) / 3
        }

        implied_prob_over = implied_odds.calculate_implied_prob(float(odds_over))
        implied_prob_under = implied_odds.calculate_implied_prob(float(odds_under))

        for model_name, pred in prediction.items():
            if model_name == 'CONSENSUS PICK':
                std_estimate = (error['regression'][2] + error['xg_boost'][2] + error['svr'][2]) / 3
            else:
                std_estimate = error[model_name][2]
            
            prob_under = float(norm.cdf(float(line), loc=pred, scale=std_estimate))
            prob_over = float(1 - prob_under)
            edge_under = prob_under - implied_prob_under    
            edge_over = prob_over - implied_prob_over
            if type(pred) == int:
                pred = [pred]

    
            # Store the prediction, standard deviation, and probabilities
            results.append({
                'Date': current_date,
                'Model': model_name,
                'Player': player_name,
                'Exact Prediction': pred[0],
                'Betting Line': line,
                'Standard Deviation Estimate (RMSE)': std_estimate,
                'Probability Over Line': f"{prob_over:.2%}",
                'Probability Under Line': f"{prob_under:.2%}",
                'Implied Probability Over (from odds)': f"{implied_prob_over:.2%}",
                'Implied Probability Under (from odds)': f"{implied_prob_under:.2%}",
                'Edge Over': edge_over,
                'Edge Under': edge_under,
                'Result': ''  # Placeholder for result
            })
    
    df_results = pd.DataFrame(results)
    current_date = datetime.today().strftime('%Y-%m-%d')
    filename = f'predictions_rebounding-{current_date}.csv'

    if os.path.exists(filename):
        df_results.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_results.to_csv(filename, index=False)
        
    print(f'Predictions saved to {filename}')



def read_csv(filename):
    df = pd.read_csv(filename)
    player_names = df['Player Name'].tolist()
    teams = df['Player Team'].tolist()
    opps = df['Opposing Team'].tolist()
    lines = df['Rebounds Line'].tolist()
    odds_overs = df['Over Odds'].tolist()
    odds_unders = df['Under Odds'].tolist()
    homes = df['Home/Away'].tolist()
    return player_names, teams, opps, lines, odds_overs, odds_unders, homes


if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_test()
    else:
        # read csv file, have file name as arg input
        player_names, teams, opps, lines, odds_overs, odds_unders, homes = read_csv(sys.argv[1])
        predict(player_names, teams, opps, lines, odds_overs, odds_unders, homes)
        # Optional, print consensus picks
        current_date = datetime.today().strftime('%Y-%m-%d')
        picks = pd.read_csv(f'predictions_rebounding-{current_date}.csv')
        # For all rows where the Date is today, save the consesnsus picks as consensus-date.csv
        consensus = picks[(picks['Model'] == 'CONSENSUS PICK') & (picks['Date'] == current_date)]
        consensus.to_csv(f'consensus-{current_date}.csv', index=False)
        print(f'Consensus picks saved to consensus-{current_date}.csv')
