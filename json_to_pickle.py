import glob
import os
import sys
import json
import pickle

def count():
  tag_freq = {}
  for e, name in enumerate(glob.glob("../metas/*.json")):
    m = json.loads(open(name).read())
    if e % 500 == 0:
      print(e, m)
    for tag in m['tags'].split(','):
      if tag_freq.get(tag) is None:
        tag_freq[tag] = 0
      tag_freq[tag] += 1
  open("tag_freq.pkl", "wb").write( pickle.dumps(tag_freq) )

def check():
  tag_freq = pickle.loads( open("tag_freq.pkl", "rb").read() )
  for e, (tag, freq) in enumerate(sorted(tag_freq.items(), key=lambda x:x[1]*-1)):
    print(e, tag, freq)
if __name__ == '__main__':
  if '--count' in sys.argv:
    count()
  if '--check' in sys.argv:
    check()
