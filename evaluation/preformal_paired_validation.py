from evaluation.preformal_part3_gates import PreformalPairedScenarioMismatchError, assert_preformal_pairable, paired_statistics, scenario_signature


def validate_paired_rows(rows, *, baseline_method, comparison_method, sensitivity=False):
    base=[r for r in rows if r.get('method_id')==baseline_method and r.get('status')=='success']
    comp=[r for r in rows if r.get('method_id')==comparison_method and r.get('status')=='success']
    bm={scenario_signature(r,sensitivity=sensitivity):r for r in base}
    cm={scenario_signature(r,sensitivity=sensitivity):r for r in comp}
    missing=[str(k) for k in set(bm)-set(cm)]
    mismatched=[]
    for k in set(bm)&set(cm):
        try: assert_preformal_pairable(bm[k],cm[k],sensitivity=sensitivity)
        except PreformalPairedScenarioMismatchError as exc: mismatched.append({'identity':str(k),'reason':str(exc)})
    status='passed' if not missing and not mismatched else 'failed'
    return {'comparison_id':f'{baseline_method}_vs_{comparison_method}','baseline_method':baseline_method,'comparison_method':comparison_method,'scenario_bank_hash':base[0].get('scenario_bank_hash') if base else None,'scenario_family_id':base[0].get('scenario_family_id') if sensitivity and base else None,'expected_paired_count':len(base),'actual_paired_count':len(set(bm)&set(cm)),'missing_pairs':missing,'mismatched_pairs':mismatched,'metric_schema':'formal_result_v2','aggregation_status':status,'publication_eligible':False}
