"""Contiguous resume selection; never silently skip an unfinished horizon."""
def select_updates(plan,rollout,completed,limit):
 rows=[r for r in plan if r['rollout']==rollout and completed<r['update']<=limit]
 if not rows or rows[0]['update']!=completed+1:
  raise ValueError('Non-contiguous resume: completed=%s rollout=%s selected=%s'%(completed,rollout,rows))
 return rows
