"""Runtime provenance report for final parameter freeze templates."""
from __future__ import annotations
from pathlib import Path
import json
from evaluation.parameter_freeze import extract_frozen_parameters, unresolved_placeholders

def build_parameter_freeze_report(config:dict, *, source_file:str)->dict:
    unresolved={p['path']:p for p in unresolved_placeholders(config)}
    rows=[]
    for p in extract_frozen_parameters(config):
        rows.append({'canonical_key':p.key,'frozen_value':p.value,'category':p.category,'source_file':source_file,'source_key':p.source,'rationale':p.rationale,'applicable_methods':'all','allowed_differences':list(p.allowed_method_differences),'resolved_status':'resolved'})
    return {'parameters':rows,'unresolved_placeholders':list(unresolved.values())}

def write_parameter_freeze_report(config:dict, output:str|Path, *, source_file:str)->dict:
    r=build_parameter_freeze_report(config, source_file=source_file); Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(r,indent=2,sort_keys=True)); return r
