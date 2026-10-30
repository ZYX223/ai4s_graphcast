"""Reject discontinuous recorded parameter/optimizer trajectories."""
def validate_trajectory(reports):
 previous=None
 for horizon,r in enumerate(sorted(reports,key=lambda x:x['rollout']),1):
  expected=list(range(1,17)) if horizon==1 else [13+2*horizon,14+2*horizon]
  if r['rollout']!=horizon or [x['update'] for x in r['rows']]!=expected:
   raise ValueError('Trajectory contains missing/repeated/out-of-stage updates')
  counters=r['checkpoint_reload']['optimizer_counters']
  if not counters or any(c!=expected[-1] for c in counters):raise ValueError('Wrong optimizer counter')
  if previous is not None:
   for field in ['data_sha256','model','task','optimizer','seed','source_revision','tiny']:
    if r[field]!=previous[field]:raise ValueError('Changed trajectory '+field)
   for initial,final in [('initial_params_sha256','params_sha256'),('initial_optimizer_sha256','optimizer_sha256')]:
    if r[initial]!=previous['checkpoint_reload'][final]:raise ValueError('Broken checkpoint digest chain: '+initial)
  previous=r
 return True
