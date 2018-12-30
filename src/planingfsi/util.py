import os


def rm_rf(dList):
 dList = map(None, dList)
 for d in dList:
   for path in (os.path.join(d,f) for f in os.listdir(d)):
     if os.path.isdir(path):
       rm_rf(path)
     else:
       os.unlink(path)
   os.rmdir(d)
