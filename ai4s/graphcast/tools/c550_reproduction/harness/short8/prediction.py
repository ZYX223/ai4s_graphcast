"""Prespecified endpoint model; never a full-course validation claim."""
from statistics import mean
from sampling_contract import validate_row,pair_spread,ROLLOUTS

def evaluate(rows):
    if len(rows) not in (8,9):raise ValueError('incomplete bounded sample')
    for i,row in enumerate(rows):validate_row(row,i+1,ROLLOUTS[i])
    t=[r['end_to_end_step_seconds'] for r in rows]
    firstpair=pair_spread(t[6:8])
    if (len(rows)==8 and firstpair>5) or (len(rows)==9 and firstpair<=5):raise ValueError('violated prespecified stop rule')
    p1=t[1:3];p6=t[4:6];p12=t[-2:];s1,s6,s12=map(mean,(p1,p6,p12))
    slope=(s12-s1)/11;intercept=s1-slope;first=t[0]-s1
    total=38*intercept+170*slope+first
    assert abs(total-(25*s1+12*s12+t[0]))<1e-8
    middle=intercept+6*slope;residual=(s6-middle)/middle*100
    spreads={str(r):pair_spread(p) for r,p in [(1,p1),(6,p6),(12,p12)]}
    gate=total>0 and min(intercept+r*slope for r in range(1,13))>0 and abs(residual)<=5 and max(spreads.values())<=5
    raw=29434.23636529152/total/132.49*100
    return dict(r1_mean_seconds=s1,r6_mean_seconds=s6,r12_mean_seconds=s12,predicted_r6_seconds=middle,
        r6_residual_pct=residual,pair_spreads_pct=spreads,r12_timed_steps=[rows[-2]['step'],rows[-1]['step']],
        first_update_excess_seconds=first,slope_seconds_per_r=slope,intercept_seconds=intercept,
        predicted_total_seconds=total,diagnostic_model_mfu_pct=raw,conditional_mfu_pct=raw if gate else None,
        conditional_h100_ratio_pct=raw/10.058595199*100 if gate else None,curve_gate_passed=gate,
        all_update_seconds=sum(t),actual_updates=len(t),first_long_pair_spread_pct=firstpair,
        omitted_third_long_update=len(rows)==8,validated_mfu_prediction=None,formal_mfu_eligible=False,
        first_r12_may_contain_switch_overhead=True,state_transfer_validated=False,sensitivity_is_confidence_interval=False)
