import requests


def run(file=None, n=0.99, k=6):

  try:
    r = requests.post(
        "https://covid19-ekfirefly.herokuapp.com/api/kmeans/",
        parameters={
          "file":file,
          "k": k,
          "n": n,
        }
      )
    return r.json()
  except Exception as e:
    return e
