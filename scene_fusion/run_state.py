"""Content-verified checkpoints and atomic, actionable run status."""
import hashlib,json,os,time
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');os.replace(temp,path)

class RunState:
    def __init__(self,out,code_hash,resume=False):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.path=self.out/'run-state.json'
        if self.path.exists() and not resume:raise ValueError('Output already contains a run. Use --resume or a new output directory.')
        self.data=json.loads(self.path.read_text(encoding='utf8')) if self.path.exists() else dict(version=1,stages={})
        self.code_hash=code_hash;self.chain='';self.data.update(status='running',error=None);self.save()
    def save(self):atomic_json(self.path,self.data)
    def stage(self,name,argv,outputs,execute):
        fingerprint=hashlib.sha256(json.dumps([self.code_hash,self.chain,argv],sort_keys=True).encode()).hexdigest()
        outputs=[Path(p) for p in outputs];old=self.data['stages'].get(name,{})
        hashes={str(p.relative_to(self.out)):digest(p) for p in outputs if p.is_file()}
        if old.get('status')=='complete' and old.get('fingerprint')==fingerprint and len(hashes)==len(outputs) and hashes==old.get('artifacts'):
            print('REUSE',name,flush=True);self.chain=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()+fingerprint.encode()).hexdigest();return
        self.data.update(stage=name,status='running');self.data['stages'][name]=dict(status='running',fingerprint=fingerprint);self.save()
        started=time.monotonic();print('START',name,flush=True)
        try:
            execute()
            missing=[str(p) for p in outputs if not p.is_file()]
            if missing:raise ValueError('Stage did not produce '+', '.join(missing))
            hashes={str(p.relative_to(self.out)):digest(p) for p in outputs}
            self.data['stages'][name]=dict(status='complete',fingerprint=fingerprint,artifacts=hashes,seconds=time.monotonic()-started)
            self.chain=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()+fingerprint.encode()).hexdigest();self.save();print('DONE',name,flush=True)
        except Exception as e:
            self.data['stages'][name].update(status='failed',error=str(e));self.data.update(status='failed',error=str(e));self.save();raise
    def complete(self):self.data.update(status='complete',stage=None);self.save()
