"""Fail-closed, persisted launch identity for restartable Modal coordinators.

A coordinator MUST receive one already prepared study as an argument; it must
never create a fresh study on re-entry. Persist intent BEFORE spawning and call
identity AFTER spawning. An interrupted ambiguous window requires investigation,
not a speculative resubmission. This does not assume retries=0 prevents platform
re-entry. Existing calls are attached, never retrained; an arm's runner still
needs its own exclusive output-directory guard.
"""
import json
from pathlib import Path


def calls_once(folder,arms,spawn,attach,commit):
    folder=Path(folder);arms=list(arms)
    if not folder.is_dir() or not arms or len(set(arms))!=len(arms) or any(not a or '/' in a or a in ('.','..') for a in arms):
        raise ValueError('existing study and unique simple arm names required')
    path=folder/'launch-ledger.json'
    def save(ledger):
        temporary=path.with_suffix('.json.pending')
        temporary.write_text(json.dumps(ledger,indent=2)+'\n');temporary.replace(path);commit()
    if path.exists():
        ledger=json.loads(path.read_text())
        if ledger.get('schema')!=1 or ledger.get('arms')!=arms or ledger.get('study_name')!=folder.name or set(ledger.get('calls',{}))-set(arms):
            raise ValueError('launch identity or arm scope drift')
    else:
        if any((folder/a).exists() for a in arms):
            raise RuntimeError('Existing arm artifacts without launch identity: investigate, do NOT spawn replacements')
        ledger=dict(schema=1,study_name=folder.name,arms=arms,calls={});save(ledger)
    calls=[]
    for arm in arms:
        if arm in ledger['calls']:
            entry=ledger['calls'][arm]
            if entry.get('state')!='spawned' or not entry.get('call_id'):
                raise RuntimeError('Ambiguous launch window for '+arm+': investigate, do NOT resubmit')
            calls.append(attach(entry['call_id']))
        else:
            if (folder/arm).exists():
                raise RuntimeError('Unrecorded existing arm '+arm+': do NOT resubmit')
            ledger['calls'][arm]=dict(state='spawning');save(ledger)
            call=spawn(arm)
            ledger['calls'][arm]=dict(state='spawned',call_id=call.object_id);save(ledger)
            calls.append(call)
    return calls
