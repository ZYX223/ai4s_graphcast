"""Persist original evaluator rows; never skip a window or alter its arithmetic."""
import ast,copy,datetime,hashlib,json,math,os
from pathlib import Path

def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()

def durable(path,value):
 path=Path(path);assert not path.exists() and not path.is_symlink(),path
 temporary=path.with_name(path.name+'.tmp')
 with temporary.open('xb') as f:
  f.write(encoded(value)+b'\n');f.flush();os.fsync(f.fileno())
 temporary.rename(path)
 fd=os.open(str(path.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def validate_row(row,index,expected):
 assert row['batch_index']==index and index<len(expected)
 assert row['valid_sample_count']==1 and row['padded_sample_count']==0
 assert len(row['lead_losses'])==12 and row['per_variable_lead_losses']
 values=[row['loss'],row['seconds'],*row['lead_losses']]
 for v in row['per_variable_lead_losses'].values():
  assert len(v)==12;values+=v
 assert all(math.isfinite(x) for x in values)
 origins=row['origins_by_rank'];assert len(origins)==1
 assert origins[0]['local_window_indices']==[expected[index]]
 assert origins[0]['sample_weights']==[1.0]

class JournalRows(list):
 def __init__(self,path,identity,expected):
  super().__init__();self.path=Path(path);self.path.mkdir(exist_ok=False)
  self.identity=copy.deepcopy(identity);self.expected=list(expected)
  assert len(expected)==111 and len(set(expected))==111
  self.previous=digest(identity)
  durable(self.path/'identity.json',dict(identity=identity,expected_indices=expected,identity_sha256=self.previous))
 def append(self,row):
  validate_row(row,len(self),self.expected)
  record=dict(identity_sha256=digest(self.identity),previous_sha256=self.previous,row=copy.deepcopy(row))
  h=digest(record);record['record_sha256']=h
  durable(self.path/('row-%04d.json'%len(self)),record)
  self.previous=h;super().append(row)
  print('validation_loss_committed',len(self),'of',len(self.expected),flush=True)

def read_prefix(path,identity):
 path=Path(path);header=json.loads((path/'identity.json').read_text());assert header['identity']==identity
 previous=digest(identity);assert header['identity_sha256']==previous
 records=[]
 for i,p in enumerate(sorted(path.glob('row-*.json'))):
  assert p.name=='row-%04d.json'%i and not p.is_symlink()
  v=json.loads(p.read_text());h=v.pop('record_sha256')
  assert digest(v)==h and v['previous_sha256']==previous and v['identity_sha256']==digest(identity)
  validate_row(v['row'],i,header['expected_indices'])
  v['record_sha256']=h;records.append(v);previous=h
 return records

def verify_complete(path,identity,result):
 rows=read_prefix(path,identity);assert len(rows)==111
 assert [v['row'] for v in rows]==result['steps']
 return dict(complete=True,committed_windows=111,identity_sha256=digest(identity),
             last_record_sha256=rows[-1]['record_sha256'],original_result_rows_unchanged=True)

def instrument(source,namespace,path,identity,expected):
 assert hashlib.sha256(source.encode()).hexdigest()==identity['runner_sha256']
 fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_validation')
 original=ast.dump(fn,include_attributes=False);patched=copy.deepcopy(fn)
 nodes=[n for n in ast.walk(patched) if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='rows']
 assert len(nodes)==1 and isinstance(nodes[0].value,ast.List) and not nodes[0].value.elts
 nodes[0].value=ast.Call(func=ast.Name(id='_cp38_journal_rows',ctx=ast.Load()),args=[],keywords=[])
 check=copy.deepcopy(patched)
 target=next(n for n in ast.walk(check) if isinstance(n,ast.Assign) and len(n.targets)==1
             and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='rows')
 target.value=ast.List(elts=[],ctx=ast.Load())
 assert ast.dump(check,include_attributes=False)==original
 rows=JournalRows(path,identity,expected)
 ns=dict(namespace);ns['_cp38_journal_rows']=lambda:rows
 tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),patched],type_ignores=[])
 exec(compile(ast.fix_missing_locations(tree),'frozen-evaluator-with-row-journal','exec'),ns)
 durable(Path(path)/'instrumentation.json',dict(original_function_ast_sha256=hashlib.sha256(original.encode()).hexdigest(),
     patched_function_ast_sha256=hashlib.sha256(ast.dump(patched,include_attributes=False).encode()).hexdigest(),
     sole_change='rows=[] replaced by append-persisting list; original evaluator arithmetic and iteration unchanged',
     utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 return ns['evaluate_validation']
