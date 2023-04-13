import argparse
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    colname = ['Ground Truth', 'Human', 'Zeroshot', 'Oneshot', 'Fifthshot']
    df = pd.read_csv(args.file_path)
    score_ar = np.array(df)[:,1:]
    score_ar = np.concatenate(np.hsplit(score_ar, args.n_question), axis = 0)
    
    avg_score = score_ar.mean(0).astype(float).round(1)
    print(avg_score)
    
    ratio = {}
    for i in range(2,5):
        odds_l = []
        for j in range(2):
            odds = (score_ar[:, i] > score_ar[:, j]).astype(float).mean().round(2)
            odds_l.append(odds)
        ratio[colname[i]] = odds_l  
    print(ratio)

    if args.plot:
        barWidth = 0.35
        plt.bar(colname, avg_score, width=barWidth)
        plt.yticks(np.arange(0, 5.1, 1))
        plt.xlabel('Avg Score( Max 5 )', fontsize = 15)
        plt.ylabel('Score', fontsize = 15)
        plt.savefig('plot/api_avg_score.png',dpi=400)
        plt.close()

        barWidth = 0.2
        br1 = np.arange(len(colname[:2]))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        p1 = plt.bar(br1, ratio[colname[2]], label=colname[:2], width=barWidth)
        p2 = plt.bar(br2, ratio[colname[3]], label=colname[:2], width=barWidth)
        p3 = plt.bar(br3, ratio[colname[4]], label=colname[:2], width=barWidth)
        plt.axhline(y = 0.5, color = 'r', linestyle = '--')
        plt.xticks([r + barWidth for r in range(len(colname[:2]))], colname[:2])
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.xlabel('Human Preference', fontsize = 15)
        plt.ylabel('Odds', fontsize = 15)
        plt.legend((p1[0], p2[0], p3[0]), ('Zero', 'One', 'Fifth'))
        
        plt.savefig('plot/api_preference.png',dpi=400)
        plt.close()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='human_preference_api.csv')
    parser.add_argument('--n_question', type=int, default=3)
    parser.add_argument('--plot', type=bool, default=True)
    args = parser.parse_args()
    main(args)
