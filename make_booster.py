
import pickle
import os
import sys
import json
import random
import glob
import re


def to_svm(arr):
  return " ".join([ "%d:%09f"%(e,a) for e, a in enumerate(arr) ])

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

  CONF = """
  objective = binary:logistic
  eta = 0.025
  gamma = 0
  min_child_weight = 1 
  max_depth = 30000
  num_round = 300
  save_period = 0 
  data = "booster_data/{train}" 
  eval[test] = "booster_data/{test}" 
  test:data = "booster_data/{test}"      
  """

  CONF = CONF.format(train="{}.train.txt".format(term), \
              test="{}.test.txt".format(term), ) 
  
  open("booster_conf/{}.conf".format(term), "w").write( CONF )  
