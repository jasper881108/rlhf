import argparse
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    colname = ['Ground Truth', 'SFT', 'PPO']
    df = pd.read_csv(args.file_path)
    score_ar = np.array(df)[:,1:]
    score_ar = np.concatenate(np.hsplit(score_ar, args.n_question), axis = 0)
    
    avg_score = score_ar.mean(0).astype(float).round(1)
    print(avg_score)
    
    ratio = {}
    for i in range(2,len(colname)):
        odds_l = []
        for j in range(2):
            odds = (score_ar[:, i] > score_ar[:, j]).astype(float).mean().round(2)
            odds_l.append(odds)
        ratio[colname[i]] = odds_l  
    print(ratio)

    if args.plot:
        barWidth = 0.35
        plt.bar(colname, avg_score, width=barWidth)
        plt.yticks(np.arange(0, 3.1, 1))
        plt.xlabel('Avg Score( Max 3 )', fontsize = 15)
        plt.ylabel('Score', fontsize = 15)
        plt.savefig('plot/ppo_avg_score.png',dpi=400)
        plt.close()

        barWidth = 0.2
        br1 = np.arange(len(colname[:2]))
        p1 = plt.bar(br1, ratio[colname[2]], label=colname[:2], width=barWidth)
        plt.axhline(y = 0.5, color = 'r', linestyle = '--')
        plt.xticks([r for r in range(len(colname[:2]))], colname[:2])
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.xlabel('Human Preference', fontsize = 15)
        plt.ylabel('Odds', fontsize = 15)
        plt.legend((p1[0],), ('PPO',))
        
        plt.savefig('plot/ppo_preference.png',dpi=400)
        plt.close()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='human_preference_ppo.csv')
    parser.add_argument('--n_question', type=int, default=3)
    parser.add_argument('--plot', type=bool, default=True)
    args = parser.parse_args()
    main(args)
