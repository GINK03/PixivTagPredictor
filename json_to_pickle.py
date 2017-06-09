import glob
import os
import sys
import json
import pickle
import re
import random
""" タグがどれだけの頻度で発生するか """
def count():
  tag_freq = {}
  for e, name in enumerate(glob.glob("../metas/*.json")):
    try:
      m = json.loads(open(name).read())
    except json.decoder.JSONDecodeError as e:
      continue
    if e % 500 == 0:
      print(e, m)
    for tag in m['tags'].split(','):
      if tag_freq.get(tag) is None:
        tag_freq[tag] = 0
      tag_freq[tag] += 1
  open("tag_freq.pkl", "wb").write( pickle.dumps(tag_freq) )

""" フレクエンシーを表示 """
def check():
  tag_freq = pickle.loads( open("tag_freq.pkl", "rb").read() )
  for e, (tag, freq) in enumerate(sorted(tag_freq.items(), key=lambda x:x[1]*-1)):
    print(e, tag, freq)

""" tag -> [(userid), (userid), (userid) ... ] """
def tag_list():
  tag_list = {}
  import re
  for e, name in enumerate(glob.glob("../metas/*.json")):
    illust_id = re.search(r"/(illust_id.*?)\.json", name).group(1)
    try:
      m = json.loads(open(name).read())
    except json.decoder.JSONDecodeError as e:
      continue
    if e % 500 == 0:
      print(e, m)
    for tag in m['tags'].split(','):
      if tag_list.get(tag) is None:
        tag_list[tag] = set()
      tag_list[tag].add( illust_id )
  open("tag_list.pkl", "wb").write( pickle.dumps(tag_list) )


"""
{ 
  userid -> vec,
  userid -> vec,
  ...
}
"""
def illustid_vec():
  illustid_vec = {}
  for ei, name in enumerate(glob.glob("/home/gimpei/vectors/*.json")):
    v        = json.loads(open(name, "r").read()) 
    illustid = re.search(r"(illust.*?)\.json", name).group(1)
    if ei%500 == 0:
      print(ei, illustid, v[:2])
    if illustid_vec.get(illustid) is None:
      illustid_vec[illustid] = v
  open("illustid_vec.pkl", "wb").write( pickle.dumps(illustid_vec) )

""" tag -> { 
  positive : {userid:vec, userid:vec, ...},
  negative : {userid:vec, userid:vec, ...}
} 
"""
def tag_pair():
  illustid_vec = pickle.loads( open("illustid_vec.pkl", "rb").read() )

  illust_ids = set()
  for e, name in enumerate(glob.glob("../metas/*.json")):
    if e%50 == 0:
      print("now loading iter", e)
    illust_id = re.search(r"/(illust_id.*?)\.json", name).group(1)
    illust_ids.add( illust_id )

  tag_list = pickle.loads( open("tag_list.pkl", "rb").read() )
  for et, (tag, positives) in enumerate(tag_list.items()):
    if len(positives) < 600:
      continue
    tag_pair = {}
    print(et, tag, len(positives) )
    
    """ ちゃんとベクタペアがあるやつ """
    pv = {}
    for p in positives:
      if illustid_vec.get(p) is None:
        continue
      pv[p] = illustid_vec[p]

    tag_pair["positive"] = pv
    """ negativeのサンプリング　"""
    negative_lot = illust_ids - positives
    negative_lot = list(negative_lot) 
    random.shuffle(negative_lot)
    
    nv = {}
    for n in negative_lot:
      if illustid_vec.get(n) is None:
        continue
      nv[n] = illustid_vec[n]
      if len(nv) >= len(pv)*10:
        break
      
    """ negativeはpositiveの10倍取る """
    tag_pair["negative"] = nv
    open("/home/gimpei/sda/tag_pair/{}.pkl".format(tag.replace("/", "_")), "wb").write( pickle.dumps(tag_pair) )
    

if __name__ == '__main__':
  if '--count' in sys.argv:
    count()
  if '--check' in sys.argv:
    check()
  if '--tag_list' in sys.argv:
    tag_list()
 
  if '--illustid_vec' in sys.argv:
    illustid_vec()

  if '--tag_pair' in sys.argv:
    tag_pair()
