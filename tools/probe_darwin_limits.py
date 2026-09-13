import os
import subprocess
import sys

code = """
import os, resource, subprocess
print(subprocess.check_output(['/bin/ps','-o','rss=,vsz=','-p',str(os.getpid())], text=True).strip(), flush=True)
try:
 resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912))
 resource.setrlimit(resource.RLIMIT_DATA,(536870912,536870912))
 print('EXACT_LIMITS_OK',flush=True)
 from PIL import Image
 import io
 b=io.BytesIO();Image.new('RGBA',(100,100),'red').save(b,'PNG');print(len(b.getvalue()),flush=True)
 print('PIL_OK',flush=True)
except Exception as exc:
 print(type(exc).__name__,str(exc),flush=True)
"""
for name, override in [('default',{}),('shared_avoid',{'DYLD_SHARED_REGION':'avoid'}),('nano_disabled',{'MallocNanoZone':'0'}),('both',{'DYLD_SHARED_REGION':'avoid','MallocNanoZone':'0'})]:
 print('VARIANT',name,flush=True)
 result=subprocess.run([sys.executable,'-c',code],env={**os.environ,**override},capture_output=True,text=True,timeout=20)
 print('returncode',result.returncode,'stdout',result.stdout,'stderr',result.stderr,flush=True)
