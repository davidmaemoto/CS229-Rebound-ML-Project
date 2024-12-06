import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

folder_path = 'predictions_data'
files = os.listdir(folder_path)

def get_consensus_accuracy():
    LIMIT = 0.13
    accuracy = {
        'regression': 0,
        'xg_boost': 0,
        'svr': 0,
        'consensus': 0
    }
    totals = {
        'regression': 0,
        'xg_boost': 0,
        'svr': 0,
        'consensus': 0
    }
    rows = 0

    all_instances = []

    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # group every 4 consecutive rows
        grouped_rows = [df.iloc[i:i+4] for i in range(0, len(df), 4)]

        for group in grouped_rows:
            if len(group) < 4:
                continue

            actual = group.iloc[0]['Rebounds']
            betting_line = group.iloc[0]['Betting Line']

            if pd.isnull(actual):
                continue

            model_preds = {}
            for _, row in group.iterrows():
                model = row['Model']
                pred = row['Exact Prediction']
                edge_over = row['Edge Over']
                edge_under = row['Edge Under']

                # Only consider this model if edges exceed LIMIT
                if (edge_over > LIMIT) or (edge_under > LIMIT):
                    if model == 'regression':
                        totals['regression'] += 1
                        if actual != betting_line:
                            correct = ((actual > betting_line and pred > betting_line) or
                                       (actual < betting_line and pred < betting_line))
                            if correct:
                                accuracy['regression'] += 1
                        model_preds['regression'] = pred

                    elif model == 'xg_boost':
                        totals['xg_boost'] += 1
                        if actual != betting_line:
                            correct = ((actual > betting_line and pred > betting_line) or
                                       (actual < betting_line and pred < betting_line))
                            if correct:
                                accuracy['xg_boost'] += 1
                        model_preds['xg_boost'] = pred

                    elif model == 'svr':
                        totals['svr'] += 1
                        if actual != betting_line:
                            correct = ((actual > betting_line and pred > betting_line) or
                                       (actual < betting_line and pred < betting_line))
                            if correct:
                                accuracy['svr'] += 1
                        model_preds['svr'] = pred

                    elif model == 'CONSENSUS PICK':
                        totals['consensus'] += 1
                        if actual != betting_line:
                            correct = ((actual > betting_line and pred > betting_line) or
                                       (actual < betting_line and pred < betting_line))
                            if correct:
                                accuracy['consensus'] += 1
                        model_preds['consensus'] = pred
            
            # Append to all_instances only if we have all three main models
            # If consensus is missing, store None
            if ('regression' in model_preds) and ('xg_boost' in model_preds) and ('svr' in model_preds):
                consensus_pred = model_preds['consensus'] if 'consensus' in model_preds else None
                all_instances.append((actual, betting_line, 
                                      model_preds['regression'], 
                                      model_preds['xg_boost'], 
                                      model_preds['svr'], 
                                      consensus_pred))
                rows += 4

    # Compute accuracy percentages
    for model in accuracy:
        if totals[model] > 0:
            accuracy[model] = accuracy[model] / totals[model]
        else:
            accuracy[model] = 0
    
    return accuracy, totals, "Total Bets: " + str(rows//4), all_instances

def find_optimal_weights(instances):
    actuals = np.array([inst[0] for inst in instances])
    reg_preds = np.array([inst[2] for inst in instances])
    xgb_preds = np.array([inst[3] for inst in instances])
    svr_preds = np.array([inst[4] for inst in instances])

    best_w = (0,0,0)
    best_error = float('inf')
    step = 0.01
    for w1 in np.arange(0, 1+step, step):
        for w2 in np.arange(0, 1+step, step):
            w3 = 1 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue
            combined_pred = w1 * reg_preds + w2 * xgb_preds + w3 * svr_preds
            mse = np.mean((actuals - combined_pred)**2)
            if mse < best_error:
                best_error = mse
                best_w = (w1, w2, w3)

    return best_w, best_error

def calculate_optimal_weight_accuracy(instances, w1, w2, w3):
    correct = 0
    total = 0
    for actual, line, reg_pred, xgb_pred, svr_pred, cons_pred in instances:
        combined_pred = w1 * reg_pred + w2 * xgb_pred + w3 * svr_pred
        if actual != line:
            total += 1
            if (actual > line and combined_pred > line) or (actual < line and combined_pred < line):
                correct += 1
    if total > 0:
        return correct / total
    else:
        return 0.0

def generate_confusion_matrix(instances, w1, w2, w3):
    """
    Generate and save confusion matrices for regression, xg_boost, svr, consensus (if available), and weighted model.
    Confusion matrix format:
    [[TP, FP],
     [FN, TN]]
    TP = Over-Over, FP = Under-Over, FN = Over-Under, TN = Under-Under
    """
    # Helper function to get confusion counts for a given model's predictions
    def get_confusion_counts(actuals, lines, preds):
        # TP, FP, FN, TN
        TP = FP = FN = TN = 0
        for a, l, p in zip(actuals, lines, preds):
            if a == l:
                # No bet scenario (tie); skip
                continue
            actual_over = a > l
            pred_over = p > l
            if actual_over and pred_over:
                TP += 1
            elif not actual_over and pred_over:
                FP += 1
            elif actual_over and not pred_over:
                FN += 1
            else: # not actual_over and not pred_over
                TN += 1
        return TP, FP, FN, TN
    
    def plot_and_save_cm(TP, FP, FN, TN, model_name):
        cm = np.array([[TP, FP],
                       [FN, TN]])
        plt.figure(figsize=(5,4))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                         xticklabels=["Over", "Under"],
                         yticklabels=["Over", "Under"])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix.png')
        plt.close()
    
    actuals = [inst[0] for inst in instances]
    lines = [inst[1] for inst in instances]
    reg_preds = [inst[2] for inst in instances]
    xgb_preds = [inst[3] for inst in instances]
    svr_preds = [inst[4] for inst in instances]
    cons_preds = [inst[5] for inst in instances]

    # Weighted model predictions
    weighted_preds = [w1*r + w2*x + w3*s for r, x, s in zip(reg_preds, xgb_preds, svr_preds)]

    # Compute confusion counts for each model
    TP, FP, FN, TN = get_confusion_counts(actuals, lines, reg_preds)
    plot_and_save_cm(TP, FP, FN, TN, 'Regression')

    TP, FP, FN, TN = get_confusion_counts(actuals, lines, xgb_preds)
    plot_and_save_cm(TP, FP, FN, TN, 'XGBoost')

    TP, FP, FN, TN = get_confusion_counts(actuals, lines, svr_preds)
    plot_and_save_cm(TP, FP, FN, TN, 'SVR')

    # Consensus may be None for some instances; filter those out
    cons_actuals = [a for (a,c) in zip(instances, cons_preds) if c is not None]
    if len(cons_actuals) == len(instances):
        # Only plot consensus if we have predictions for all instances
        # or handle partial predictions by filtering arrays
        actuals_cons = []
        lines_cons = []
        preds_cons = []
        for (a, l, _, _, _, c) in instances:
            if c is not None:
                actuals_cons.append(a)
                lines_cons.append(l)
                preds_cons.append(c)
        TP, FP, FN, TN = get_confusion_counts(actuals_cons, lines_cons, preds_cons)
        plot_and_save_cm(TP, FP, FN, TN, 'Consensus')
    
    # Weighted model
    TP, FP, FN, TN = get_confusion_counts(actuals, lines, weighted_preds)
    plot_and_save_cm(TP, FP, FN, TN, 'Weighted')


if __name__ == "__main__":
    print("Calculating consensus accuracy...")
    acc, totals, rows, instances = get_consensus_accuracy()
    print("Model Accuracies:", acc)
    print("Totals:", totals)
    print(rows)

    print("Finding optimal weights...")
    w, mse = find_optimal_weights(instances)
    print("Optimal weights:", w, "with MSE:", mse)

    opt_weight_accuracy = calculate_optimal_weight_accuracy(instances, *w)
    print("Optimal Weighted Model Accuracy:", opt_weight_accuracy)

    print("Generating confusion matrices...")
    generate_confusion_matrix(instances, *w)
    print("Confusion matrices saved.")
