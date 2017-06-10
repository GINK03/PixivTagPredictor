
import pickle
import os
import sys
import json
import random
import glob
import re

import xgboost as xgb
import os.path


def to_svm(arr):
  return " ".join([ "%d:%09f"%(e,a) for e, a in enumerate(arr) ])

PATH = "/home/gimpei/{}/sda/tag_pair/*.pkl".format( "6" )

def train():
  files = glob.glob(PATH)
  random.shuffle( files ) 
  for name in files:
    """ すでにやったやつは飛ばす """ 
    term = re.search(r"(.*?)\.pkl", name.split("/").pop()).group(1)
    save_name = "booster_models/{}.model".format(term)
    if os.path.isfile(save_name):
      print("already proceeded", term)
      continue

    """ 今回は艦これ以外やらない """
    #if "艦隊これくしょん" not in term and "艦これ" not in term:
    #  continue
    print("deal to", term)

    try:
      pair = pickle.loads( open(name, "rb").read() )
    except EOFError as e:
      print(e)
      continue

    ps = list(map(lambda x:"1 " + to_svm(x), pair["positive"].values()))

    ns = list(map(lambda x:"0 " + to_svm(x), pair["negative"].values()))

    ts = ps + ns

    random.shuffle(ts)

    tl = len(ts)
    # 50000より大きいのは見ない
    if tl > 50000: 
      print(tl, "skipします")
      continue
    print(term)
    dtrain  = "\n".join( ts[:int(tl*0.8)] )
    dtest   = "\n".join( ts[int(tl*0.8):] )
    open("booster_data/{}.train.txt".format(term), "w").write( dtrain )  
    open("booster_data/{}.test.txt".format(term), "w").write( dtest )  
    print(term, len(ts) )

    """ train & save models """
    try:
      dtrain    = xgb.DMatrix( "booster_data/{}.train.txt".format(term) )
      dtest     = xgb.DMatrix( "booster_data/{}.test.txt".format(term) )
      param     = {'max_depth':1000, 'eta':0.025, 'silent':1, 'objective':'binary:logistic' }
      num_round = 300
      evallist  = [(dtest,'eval'), (dtrain,'train')]
      bst       = xgb.train( param, dtrain, num_round, evallist )
      bst.save_model(save_name)
    except xgb.core.XGBoostError as e:
      print(e)
      continue

    for ps in bst.predict(dtest):
      print(ps)

if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
