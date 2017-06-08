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
  for ei, name in enumerate(glob.glob("/home/gimpei/19/sdb/keras-tiny-vae/vectors/*.json")):
    v        = json.loads(open(name, "r").read()) 
    illustid = re.search(r"(illust.*?)\.json", name).group(1)
    if ei%500 == 0:
      print(ei, illustid, v[:2])
    if illustid_vec.get(illustid) is None:
      illustid_vec[illustid] = v
  open("illustid_vec.pkl", "wb").write( pickle.dumps(illustid_vec) )

""" tag -> { 
  positive : [(userid), (userid), ...],
  negative : [(userid), (userid), ...]
} 
"""
def tag_pair():
  tag_pair = {}
  illust_ids = set()
  for e, name in enumerate(glob.glob("../metas/*.json")):
    illust_id = re.search(r"/(illust_id.*?)\.json", name).group(1)
    illust_ids.add( illust_id )

  tag_list = pickle.loads( open("tag_list.pkl", "rb").read() )
  for et, (tag, positives) in enumerate(tag_list.items()):
    if len(positives) < 600:
      continue
    print(et, tag, len(positives) )
    tag_pair[tag] = {}
    tag_pair[tag]["positive"] = positives
    """ negativeのサンプリング　"""
    negative_lot = illust_ids - positives
    negative_lot = list(negative_lot) 
    random.shuffle(negative_lot)
    """ negativeはpositiveの５倍取る """
    negatives   = negative_lot[:len(positives) * 5]
    tag_pair[tag]["negative"] = negatives

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
