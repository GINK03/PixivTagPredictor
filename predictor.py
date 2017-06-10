
import pickle
import os
import sys
import json
import random
import glob
import re

import xgboost as xgb
import os.path


PATH = "/home/gimpei/{}/sda/tag_pair/*.pkl".format( "6" )

""" モデルをロード """
name_model = {}
def load_models():
  for model in glob.glob("booster_models/*.model"):
    print(model)
    name = re.search(r"/(.*?)\.model", model).group(1)
    bst = xgb.Booster({'nthread':4}) #init model
    bst.load_model(model) # load data
    name_model[name] = bst

""" 幾つかテスト """
def test():
  illustid_vec = pickle.loads( open("illustid_vec.pkl", "rb").read() )
  
  illustid_name_prob = {}
  deals = [(i,v) for i,v in illustid_vec.items()]
  size  = len(deals)
  random.shuffle(deals)
  for ed, (illustid, vec) in enumerate(deals):
    print(illustid, ed, "/", size)
    name_prob = {}
    for name, model in name_model.items():
      label = [i for i,v in enumerate(vec)]
      data  = xgb.DMatrix( [vec], label=label )

      p = model.predict(data).tolist().pop(0)
      name_prob[name] = p
    print(name_prob)
    illustid_name_prob[illustid] = name_prob 
  open("illustid_name_prob.pkl", "wb").write( pickle.dumps(illustid_name_prob) )

load_models()
test()
sys.exit()



if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
