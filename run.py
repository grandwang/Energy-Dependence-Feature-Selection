# SPDX-License-Identifier: GPL-3.0-only
"""Minimal EDFS reproduction: three datasets, one seed/ratio, three budgets."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import importlib.metadata
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from data_utils import ROOT, read_json, load_dataset, sha
from edfs import EDFS

def score(y, prediction):
    return dict(accuracy=float(accuracy_score(y, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, average='macro', zero_division=0)),
                weighted_f1=float(f1_score(y, prediction, average='weighted', zero_division=0)))

def check_partition(z, n):
    arrays=[]
    for key in ('train_indices','test_indices','labeled_train_indices','unlabeled_train_indices'):
        v=z[key]
        if v.ndim != 1 or v.dtype.kind not in 'iu' or len(np.unique(v)) != len(v) or not len(v):
            raise ValueError('Invalid split indices: '+key)
        arrays.append(v.astype(int))
    tr, te, lab, unlab=arrays
    if set(tr)&set(te) or set(tr)|set(te)!=set(range(n)):
        raise ValueError('Invalid outer partition')
    if set(lab)&set(unlab) or set(lab)|set(unlab)!=set(range(len(tr))):
        raise ValueError('Invalid labeled/unlabeled partition')
    return arrays

def smoke():
    results=[]
    for item in read_json(ROOT/'reference/fixtures.json'):
        with np.load(ROOT/item['file'], allow_pickle=False) as z:
            model=EDFS().fit(z['X_l'],z['y_l'],z['X_u'])
            order=model.select(item['budget'])
            if not np.array_equal(order,z['order']):
                raise AssertionError(item['id']+': selection differs')
            actual=dict(relevance=model.relevance_,shrinkage=np.array([model.shrinkage_]),
                        correlation=np.column_stack([model._correlation_column(int(j)) for j in order]),
                        objective=np.array(model._objective_values),gains=np.array(model._marginal_gains))
            errors={}
            for key,value in actual.items():
                np.testing.assert_allclose(value,z[key],atol=1e-12,rtol=0,equal_nan=False)
                errors[key]=float(np.max(np.abs(value-z[key])))
            for k in (max(1,item['budget']//2),item['budget']):
                fresh=EDFS().fit(z['X_l'],z['y_l'],z['X_u']).select(k)
                np.testing.assert_array_equal(fresh,z['order'][:k])
            results.append(dict(fixture=item['id'],status='PASS',max_absolute_errors=errors))
    return dict(scope='smoke',status='PASS',cases=results), 0

def benchmark(data_dir):
    protocol=read_json(ROOT/'protocol.json')
    datasets={d['dataset']:d for d in read_json(ROOT/'datasets.json')}
    expected=read_json(ROOT/'reference/expected.json')
    records=[]
    for task in protocol['tasks']:
        item=datasets[task['dataset']]
        X,y=load_dataset(data_dir/item['file'],item)
        split=ROOT/task['file']
        if sha(split)!=task['sha256']:
            raise ValueError('Split hash mismatch')
        with np.load(split,allow_pickle=False) as z:
            tr,te,lab,unlab=check_partition(z,len(X))
        raw=X[tr]; mean=raw.mean(axis=0); scale=raw.std(axis=0)
        scale[scale<1e-12]=1
        train=(raw-mean)/scale; test=(X[te]-mean)/scale
        model=EDFS().fit(train[lab],y[tr][lab],train[unlab])
        order=model.select(max(protocol['budgets']))
        for budget in protocol['budgets']:
            chosen=order[:budget]
            clf=KNeighborsClassifier(n_neighbors=3)
            # All outer-training labels are used downstream, as in the source protocol.
            clf.fit(train[:,chosen],y[tr])
            values=score(y[te],clf.predict(test[:,chosen]))
            refs=[r for r in expected if r['dataset']==item['dataset'] and r['seed']==task['seed']
                  and r['label_ratio']==task['label_ratio'] and r['feature_budget']==budget]
            if len(refs)!=1 or refs[0]['split_sha256']!=task['sha256']:
                raise ValueError('Missing, duplicate, or mismatched reference')
            ref=refs[0]
            deltas={k:abs(v-ref['metrics'][k]) for k,v in values.items()}
            exact=chosen.tolist()==ref['selected_features']
            records.append(dict(dataset=item['dataset'],budget=budget,selected_features=chosen.tolist(),
                                metrics=values,order_exact=exact,metric_absolute_differences=deltas,
                                historical_fields_match=exact and all(v<=1e-14 for v in deltas.values())))
        print('Completed:',item['dataset'],flush=True)
    passed=len(records)==9 and all(r['historical_fields_match'] for r in records)
    return dict(scope='benchmark',status='PASS' if passed else 'DIFFERENT',records=records,
                verification_scope='Selected prefixes and historical 3-NN metrics only; not full-paper reproduction.',
                historical_predictions_and_process_arrays='NOT_AVAILABLE'), 0 if passed else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['smoke','benchmark'],default='smoke')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    args=parser.parse_args()
    out=args.output_dir/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    try:
        with threadpool_limits(limits=1):
            result,code=smoke() if args.scope=='smoke' else benchmark(args.data_dir)
    except Exception as error:
        result=dict(scope=args.scope,status='ERROR',error=str(error));code=2
    result['environment']=dict(python=platform.python_version(),platform=platform.platform(),
                              packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','joblib','threadpoolctl']})
    result['exit_code']=code
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(result['status'],out/'results.json')
    if 'error' in result: print(result['error'],file=sys.stderr)
    return code

if __name__=='__main__':
    sys.exit(main())
