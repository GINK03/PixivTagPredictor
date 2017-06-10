
import pickle

tag_list = pickle.loads( open("./tag_list.pkl", "rb").read() )
for tag, l in tag_list.items():
  try:
    if tag[0] == "電":
      print(tag, l)
  except :
    continue
