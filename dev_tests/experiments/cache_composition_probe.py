"""Compare full cached source with caller-selected records and deterministic math."""
import argparse
import json
import secrets
from pathlib import Path

from dev_tests.experiments import prompt_cache_probe as p


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--model',required=True)
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args()
    docs,_=p.build_corpus(200)
    salt='operands-'+secrets.token_hex(5)
    calls=[]
    for variant in ['full_cached','selected_records']:
        for doc_key,seed in [('alpha_v1','alpha'),('malformed_v1','gamma')]:
            records=p._records(seed,200)
            for count in [1,2]:
                ids=['REC-0100','REC-0133'][:count]
                expected={'records':[{'record_id':r['id'],'qty':r['qty'],'unit_price_cents':r['unit_price_cents']} for r in records if r['id'] in ids]}
                question=f'Extract qty and unit_price_cents for record IDs {ids}, in that order. Return only JSON with shape {{"records":[{{"record_id":"...","qty":0,"unit_price_cents":0}}]}}.'
                text=docs[doc_key]
                if variant=='selected_records':
                    # Caller already knows the IDs; exact structured-record selection, not semantic search.
                    text='\n'.join(line for line in text.splitlines() if line.split(' | ',1)[0] in ids)
                messages,stable=p.build_messages('user_user_cc' if variant=='full_cached' else 'user_user',salt+'/'+variant,text,question)
                step=p.Step(doc_key,f'operands_{count}','', 'json', expected)
                body={'model':args.model,'messages':messages,'max_tokens':256,'reasoning':{'effort':'none'},'transforms':[],'provider':{'order':['OpenAI' if args.model.startswith('openai/') else 'Anthropic'],'allow_fallbacks':False}}
                bound=p.input_token_bound(messages)
                cfg=p.Config(model=args.model,variants=('user_user_cc',),run_id=salt,max_output_tokens=256,input_usd_per_m=0.15,cache_write_usd_per_m=0.2,output_usd_per_m=1)
                calls.append(p.PlannedCall(len(calls)+1,variant,step,body,p.sha256_json(body),p.sha256_json(stable),count==2 and variant=='full_cached',bound,p.reserve_usd(bound,cfg)))
    journal=p.Journal(args.output/(salt+'.jsonl'))
    journal.write({'type':'run_start','scope':'Synthetic extraction followed by Python arithmetic; raw transport only','model':args.model,'planned':len(calls),'max_liability_usd':sum(c.reserve_usd for c in calls),'cases':[{'index':c.index,'expected':c.step.expected,'computed_expected':sum(r['qty']*r['unit_price_cents'] for r in c.step.expected['records'])} for c in calls]})
    sender=p.OpenRouterSender(60)
    try:
        summary=p.execute(calls,send=sender,journal=journal,budget_usd=2,max_output_tokens=256,evidence='paid_live')
        journal.write({'type':'run_end',**summary})
    finally:
        sender.close();journal.close()
    # Deterministic computation only after all operands pass the source-grounded exact check.
    computations=[]
    for line in journal.path.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row.get('type')=='result' and row.get('check',{}).get('correct'):
            value=json.loads(row['content'])
            computations.append({'call':row['call'],'sum_products':sum(r['qty']*r['unit_price_cents'] for r in value['records'])})
    summary['validated_computations']=computations
    summary['journal']=str(journal.path)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
