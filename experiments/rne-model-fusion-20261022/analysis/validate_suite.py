import subprocess,sys
commands=[
 ['profile_summary.py','/results/profile-baseline-r1'],
 ['profile_summary.py','/results/profile-fused-r1','--fused'],
 ['hlo_audit.py','/results/profile-fused-r1','--policy','/results/experiment-r1/policy.json'],
 ['compare_runs.py','--baseline','/results/bench-baseline-r1','--candidate','/results/bench-fused-r1','--steps','40','--output','/results/40-step-comparison.json'],
 ['repeatability.py'],
]
for script,*args in commands:
 print('VALIDATE',script,flush=True)
 subprocess.run([sys.executable,'/results/analysis/'+script]+args,check=True)
