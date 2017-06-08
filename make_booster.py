
import pickle
import os
import sys
import json
import random
import glob
import re

import xgboost as xgb

def to_svm(arr):
  return " ".join([ "%d:%09f"%(e,a) for e, a in enumerate(arr) ])


def train():
  for name in glob.glob("tag_pair/*.pkl"):
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
    term = re.search(r"/(.*?)\.pkl", name).group(1)
    dtrain  = "\n".join( ts[:int(tl*0.8)] )
    dtest   = "\n".join( ts[int(tl*0.8):] )
    open("booster_data/{}.train.txt".format(term), "w").write( dtrain )  
    open("booster_data/{}.test.txt".format(term), "w").write( dtrain )  
    print(term, len(ts) )

    """ train & save models """
    save_name = "booster_models/{}.model".format(term)
    dtrain    = xgb.DMatrix( "booster_data/{}.train.txt".format(term) )
    dtest     = xgb.DMatrix( "booster_data/{}.test.txt".format(term) )
    param     = {'max_depth':1000, 'eta':0.025, 'silent':1, 'objective':'binary:logistic' }
    num_round = 300
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    bst       = xgb.train( param, dtrain, num_round, evallist )
    bst.dump_model(save_name)

    for ps in bst.predict(dtest):
      print(ps)

if __name__ == '__main__':
  if '--train' in sys.argv:
    train()