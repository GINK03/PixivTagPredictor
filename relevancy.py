import pickle
import os
import sys
import glob
import math
import functools 
def chaine():
  illustid_vec = pickle.loads( open("illustid_vec.pkl", "rb").read() ) 
  while True:
    print("キーを入力してください")
    illustid_ = input()
    if illustid_vec.get(illustid_) is None:
      print("キーが存在しません、再度お願いします")

    else: 
      base = illustid_vec[illustid_]

      illustid_score = {}
      for ei, (illustid, vec) in enumerate(illustid_vec.items()):
        
        #head = sum( map(lambda x: x[0]*x[1], zip(base, vec) ) )
        #tail = (sum(map(lambda x:x**2, base))**0.5 ) * (sum(map(lambda x:x**2,vec))**0.5)
        #score = head/tail
        score = functools.reduce(lambda y,x:y+x, map(lambda x: abs( x[0] - x[1] ), zip(base, vec) ) )
        if ei%100 == 0:
          print("now iter ", ei, "score", score)
        illustid_score[illustid] = score

      open("{}.pkl".format(illustid_), "wb").write( pickle.dumps(illustid_score) )


def check():
  for name in glob.glob("check/*.pkl"):
    iid_score = pickle.loads( open(name, "rb").read() )
    for iid, score in sorted(iid_score.items(), key=lambda x:x[1])[:124]:
      print(name, "https://www.pixiv.net/member_illust.php?mode=medium&%s"%iid, score)


if __name__ == '__main__':
  if '--chaine' in sys.argv:
    chaine()

  if '--check' in sys.argv:
    check()    
