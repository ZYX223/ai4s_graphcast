import copy

def require_same_workload(a,b,source_a,source_b):
    a,b=copy.deepcopy(a),copy.deepcopy(b)
    for contract,source in [(a,source_a),(b,source_b)]:
        runtime=contract['operator_replacement']['runtime']
        assert runtime['flaggems_source_root']==source
        assert runtime['flaggems_module_origin']==source+'/src/flag_gems/__init__.py'
        runtime['flaggems_source_root']='<verified-isolated-source>'
        runtime['flaggems_module_origin']='<verified-isolated-source>/src/flag_gems/__init__.py'
    assert a==b, 'workload differs beyond explicitly bound source paths'
