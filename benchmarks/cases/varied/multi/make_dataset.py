import json, random
from pathlib import Path

OUT=Path(__file__).with_name('train-120.jsonl')
rng=random.Random(120923)
domains=[
 ('community clinics','vaccine cooler','specimen courier','appointment portal','sterilizer','staff rota','lab analyzer','prescription queue','patient shuttle','intake scanner','oxygen manifold'),
 ('credit unions','member transfer','cash recycler','statement service','branch key system','loan document queue','card authorization','ATM network','settlement batch','account recovery','fraud review'),
 ('regional transit','platform lift','signal cabinet','fare validator','vehicle dispatch','timetable feed','station radio','track inspection','passenger display','crew roster','maintenance depot'),
 ('public libraries','book return belt','catalog search','branch lift','loan kiosk','digital archive','reading-room HVAC','delivery van','membership portal','large-print reader','event booking'),
 ('coastal research','tidal gauge','sample freezer','survey vessel','field radio','habitat camera','water probe','permit register','buoy battery','specimen transfer','shore station'),
 ('food cooperatives','cold room','produce scale','supplier portal','delivery route','allergen labeler','checkout terminal','warehouse lift','inventory feed','packing line','farm pickup'),
 ('municipal permitting','plan-review portal','records scanner','inspection calendar','fee ledger','public notice board','document archive','counter queue','field tablet','address registry','permit printer'),
 ('museum conservation','gallery humidity system','loan crate','condition-imaging station','security sensor','collection database','loading lift','conservation freezer','visitor ticketing','lighting controller','artifact courier'),
 ('small manufacturers','line controller','parts conveyor','pressure vessel','quality database','shift handoff board','tool crib','packing station','robot cell','power monitor','supplier gateway'),
 ('community energy','feeder monitor','substation relay','billing portal','crew dispatch','battery bank','meter gateway','outage map','transformer store','field radio','solar inverter'),
 ('regional farms','irrigation pump','grain dryer','livestock gate','weather station','milk chiller','seed inventory','field tablet','packing shed','water sampler','tractor fleet'),
 ('arts colleges','studio kiln','course registration','campus shuttle','accessibility lift','equipment checkout','student records portal','stage rigging','media archive','dorm boiler','research lab'),
]
incidents=[
 ('reported intermittent readings after a scheduled restart','a separate calibrated instrument stayed within range','the technician retained the restart log and calibration certificate'),
 ('stopped accepting a subset of valid requests during a supplier maintenance window','a documented manual route can handle one shift','the service desk recorded affected requests and the supplier window'),
 ('showed an unexpected access attempt against a restricted workspace','the access gateway blocked the request and logs show no successful reads','the security analyst preserved gateway events and the policy version'),
 ('developed a mechanical fault during routine operation','an inspected backup unit is available nearby, but transfer takes until tomorrow','the shift lead logged the fault and checked the backup inventory'),
 ('produced conflicting totals between two daily reports','the underlying transactions reconcile and no funds or records are missing','both report versions and the reconciliation worksheet are retained'),
 ('experienced a short interruption while a configuration was being deployed','rollback restored normal operation and no dependent work was lost','the change ticket records approver, timestamps, and rollback result'),
 ('received a complaint about a delayed handoff','the recipient confirmed arrival later that day and no safety or deadline threshold was missed','the handoff ledger and recipient confirmation are available'),
 ('flagged a possible data mismatch in one exported file','the file stayed in a restricted review folder and a checksum verifies the source copy','the export manifest and reviewer notes identify the file version'),
 ('lost a noncritical display feature during a network fluctuation','core service remains available and a tested local fallback is working','the operations log contains the outage interval and fallback check'),
 ('showed wear during a planned inspection before any breakdown','a compatible replacement is in stock and the component remains within its operating limit','the inspection images and maintenance record are linked'),
]
# Rephrase facts per domain and row while preserving explicit, independently grounded evidence.
contexts=[
 'The {asset} {incident}; {mitigation}. {records}. The duty coordinator is on site, and the next scheduled review is in two hours.',
 'During the morning shift, the {asset} {incident}. {mitigation}. {records}. No other linked service has reported a fault.',
 'A fictional {asset} case was opened after it {incident}. {mitigation}; {records}. The operator has authority to pause only this workstream.',
 'At a routine handover, staff noted that the {asset} {incident}. {mitigation}. {records}. A second operator can verify any corrective change.',
]
actions=[
 ('isolate','Pause the affected component, preserve the current evidence, and verify scope before resuming.'),
 ('workaround','Use the documented bounded fallback while the confirmed repair window is active.'),
 ('monitor','Keep the service in operation with a named owner and a time-boxed check against the stated limits.'),
 ('escalate','Route the unresolved condition to the accountable specialist with the retained records.'),
 ('close','Record the verified normal condition and close the item without further operational change.'),
]
# Each scenario includes an independently stated urgency/evidence posture; action target keyed from urgency.
labels=['minimal','partial','adequate','strong','complete']
levels=[
 {'label':'minimal','criterion':'Available notes do not link the event to the decision, accountable role, or any verifiable follow-up.','value':0},
 {'label':'partial','criterion':'A basic note exists, but source linkage or an important timestamp remains unclear.','value':1},
 {'label':'adequate','criterion':'The principal event and action are recorded, with one corroborating source or minor gap.','value':2},
 {'label':'strong','criterion':'The event, decision, owner, timing, and independent corroboration are linked.','value':3},
 {'label':'complete','criterion':'A reviewable chronology links sources, decision authority, action, verification, and follow-up.','value':4},
]
records=[]
for di,entry in enumerate(domains):
 domain,assets=entry[0],entry[1:]
 for j,asset in enumerate(assets):
  idx=di*10+j
  incident,mitigation,records_fact=incidents[(j+di)%len(incidents)]
  urgency=(idx*7+di+j)%5 # 0 routine, 1 bounded, 2 imminent, 3 confirmed active, 4 safety/security
  posture=[
   'The operator confirms there is no active service interruption or exposure.',
   'The issue is contained; no deadline, safety limit, or protected record is currently affected.',
   'A required handoff is due before the next shift, though a staffed fallback remains available.',
   'The current condition is confirmed and could affect a protected service if left through this shift.',
   'A verified safety or restricted-access control is currently failing and the affected activity is underway.'
  ][urgency]
  completeness=(idx*3+j+di)%5
  record_state={'service_area':domain,'asset':asset,'incident_note':contexts[idx%len(contexts)].format(asset=asset,incident=incident,mitigation=mitigation,records=records_fact), 'operational_status':posture,'record_quality_note':[
   'Available notes do not link the event to its decision owner or any verifiable follow-up.',
   'A brief shift note exists, but its source linkage or one important timestamp remains unclear.',
   'The event and response are documented and one independent source is attached.',
   'A named owner, timestamps, source evidence, and verification are linked.',
   'A complete chronology includes approvals, source references, verification, and the next check.'
  ][completeness]}
  # Action depends on facts but varies by urgency and available containment.
  action_i=[4,2,3,0,0][urgency]
  selected=actions[action_i][0]
  # Generate a distinct 2-5 candidate set and shuffle so no answer key convention exists.
  n=2+(idx%4); pool=actions.copy(); rng.shuffle(pool); chosen=pool[:n]
  if not any(x[0]==selected for x in chosen): chosen[-1]=actions[action_i]; rng.shuffle(chosen)
  opts={f'route_{k+1}_{(idx*11+k)%97}':x[1] for k,x in enumerate(chosen)}
  key=list(opts)[next(k for k,x in enumerate(chosen) if x[0]==selected)]
  choice_prompts=[f'What is the most proportionate next operational step for this {asset} case, given the documented controls and condition?',f'Which response best fits the evidence and authority described for the {asset}?',f'Choose the next action that protects continuity while respecting the stated scope for this {asset}.']
  qchoice={'type':'choice','prompt':choice_prompts[idx%3],'options':opts}
  qnoul_prompts=[f'Do the stated facts call for an immediate protective intervention on this {asset}, rather than the ordinary scheduled follow-up?',f'Is a same-shift response required for the {asset} based on the active condition and limits described?',f'Would waiting for the next routine review leave a currently confirmed safety, access, or service risk unaddressed for this {asset}?']
  qnoul={'type':'noul','prompt':qnoul_prompts[(idx//2)%3]}; booltarget=urgency>=3
  nlevels=3+(idx%3); rubric=levels[:nlevels]; levelidx=min(nlevels-1,round(completeness*(nlevels-1)/4)); ltarget=rubric[levelidx]
  qscore={'type':'score','prompt':f'Rate the auditability of this {asset} record: how well can an independent reviewer reconstruct the event, decision, evidence, and follow-up?', 'levels':rubric}
  # Rotate family IDs within allowed range independently for all three task slots.
  family=lambda task:f'qf-{6+3*((idx+di)%3)+task:08d}'
  records.append({'id':f'mq-train-{idx+1:03d}','domain':domain,'request':{'state':record_state,'questions':{'response_plan':qchoice,'immediate_intervention':qnoul,'record_auditability':qscore}},'reference':{'response_plan':{'target':{'selected':key},'rationale':f'{actions[action_i][1]} The active status and verified fallback or risk in the state make this more proportionate than the alternatives.','question_family_id':family(0)},'immediate_intervention':{'target':{'value':booltarget},'rationale':f'An immediate intervention is warranted only where the state confirms an active safety, access, or service risk that routine review would leave exposed; here the stated operational status is: {posture}','question_family_id':family(1)},'record_auditability':{'target':{'label':ltarget['label'],'value':ltarget['value']},'rationale':f'The record supports the {ltarget["label"]} level because {ltarget["criterion"].lower()} The state lists this record posture explicitly without naming a score.','question_family_id':family(2)}}})
with OUT.open('w',encoding='utf-8') as f:
 for r in records:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
print(f'wrote {len(records)} cases to {OUT}')
