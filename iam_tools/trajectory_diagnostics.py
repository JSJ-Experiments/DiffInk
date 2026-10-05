"""CPU-only trajectory/pen autopsy, excluding all padding and display-only changes."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from .preview import render
from .preprocess import sequence_strokes

LABELS=['continue','pen_up','end_char']


def diagnostics(prediction,truth,scale):
    prediction=np.asarray(prediction);truth=np.asarray(truth)
    if prediction.shape!=truth.shape or truth.ndim!=2 or truth.shape[1]!=5 or not len(truth):
        raise ValueError('expected equal nonempty unpadded N x 5 sequences')
    if not (np.isfinite(prediction).all() and np.isfinite(truth).all()) or scale<=0:raise ValueError('nonfinite or invalid scale')
    if not np.all(truth[:,2:].sum(1)==1) or not np.isin(truth[:,2:],[0,1]).all():raise ValueError('invalid ground truth states')
    xy=prediction[:,:2];target=truth[:,:2]*scale
    axes={}
    for i,name in enumerate(['x','y']):
        a,b=xy[:,i],target[:,i]
        corr=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else None
        axes[name]={'rmse_model_units':float(np.sqrt(np.mean((a-b)**2))),
                    'correlation':corr,'pred_std_model_units':float(np.std(a)),
                    'true_std_model_units':float(np.std(b))}
    true=truth[:,2:].argmax(1);pred=prediction[:,2:].argmax(1)
    confusion=np.zeros((3,3),dtype=int);np.add.at(confusion,(true,pred),1)
    counts=confusion.sum(1);predcounts=confusion.sum(0)
    return {'points':len(truth),'axes':axes,'state_labels':LABELS,
            'true_counts':counts.tolist(),'predicted_counts':predcounts.tolist(),
            'confusion_true_rows_pred_columns':confusion.tolist(),
            'per_class_recall':[(float(confusion[i,i]/n) if n else None) for i,n in enumerate(counts)],
            'pen_accuracy':float(np.trace(confusion)/len(truth)),
            'inverse_frequency_weights_real_only':[(float(len(truth)/n) if n else 0.0) for n in counts]}


def render_seq(sequence,destination,sample):
    seq=np.asarray(sequence).copy();seq[-1,2:]=[0,0,1] # display-only endpoint, never saved over raw prediction
    strokes=sequence_strokes(seq)
    for stroke in strokes:stroke[:,1]*=-1
    render({'id':Path(destination).name,'writer_id':sample['writer_id'],'text':sample['text'],
            'strokes':[s.tolist() for s in strokes]},destination)


def save_views(directory,name,prediction,truth,scale,sample):
    directory=Path(directory);prediction=np.asarray(prediction)
    if prediction.shape!=truth.shape:raise ValueError('prediction/truth mismatch')
    pred=prediction.copy();pred[:,:2]/=scale
    teacher=pred.copy();teacher[:,2:]=truth[:,2:]
    np.save(directory/f'{name}-prediction-model-units.npy',prediction)
    render_seq(pred,directory/f'{name}-predicted-pen',sample)
    render_seq(teacher,directory/f'{name}-gt-pen',sample)
    render_seq(truth,directory/f'{name}-input',sample)
    # Compare all views in a common data-coordinate frame, not separately autoscaled.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(15,4.5),sharex=True,sharey=True)
    for ax,seq,label in zip(axes,[truth,teacher,pred],['input','predicted XY + true pen','predicted XY + predicted pen']):
        display=seq.copy();display[-1,2:]=[0,0,1]
        for stroke in sequence_strokes(display):
            ax.plot(stroke[:,0],stroke[:,1],color='black',linewidth=.7,marker='.',markersize=1)
        ax.set_aspect('equal',adjustable='box');ax.set_title(label);ax.axis('off')
    fig.suptitle(sample['text']);fig.tight_layout();fig.savefig(directory/f'{name}-comparison.png',dpi=120);plt.close(fig)
    counts=diagnostics(prediction,truth,scale)
    fig,ax=plt.subplots(figsize=(6,3))
    positions=np.arange(3)
    ax.bar(positions-.18,counts['true_counts'],width=.36,label='true')
    ax.bar(positions+.18,counts['predicted_counts'],width=.36,label='predicted')
    ax.set_xticks(positions,LABELS);ax.set_ylabel('real points (padding excluded)');ax.legend()
    fig.tight_layout();fig.savefig(directory/f'{name}-pen-histogram.png',dpi=120);plt.close(fig)



def audit_existing(directory,data_root):
    directory=Path(directory);root=Path(data_root)
    rows=[json.loads(x) for x in (directory/'fixed_metrics.jsonl').read_text().splitlines()]
    scale=.01
    result_path=directory/'checkpoint.pt'
    # Read the recorded adapter without allocating GPU or loading model weights.
    if (directory/'result.json').exists():
        result=json.loads((directory/'result.json').read_text())
        if result.get('model_input_scale') is not None:scale=result['model_input_scale']
    # This older run recorded the scale only in its checkpoint.
    if result_path.exists():
        import torch
        checkpoint=torch.load(result_path,map_location='cpu',weights_only=True)
        scale=checkpoint['config']['model_input_scale'];del checkpoint
    records=[]
    for row in rows:
        name=f"{row['split']}-step-{row['step']}"
        with h5py.File(root/('tiny_train.h5' if row['split']=='train' else 'tiny_val.h5'),'r') as hf:
            truth=hf[row['sample_id']]['point_seq'][:]
        prediction=np.load(directory/f'{name}-prediction-model-units.npy')
        report=diagnostics(prediction,truth,scale)
        padded=(len(truth)+7)//8*8
        padded_counts=report['true_counts'].copy();padded_counts[2]+=padded-len(truth)
        report['legacy_weights_fixed_single_sample_with_padding']=[padded/n if n else 0 for n in padded_counts]
        report.update(step=row['step'],split=row['split'],sample_id=row['sample_id'],model_input_scale=scale)
        save_views(directory,name,prediction,truth,scale,row);records.append(report)
    output={'gpu':False,'training':False,'new_optimizer_steps':0,'padding_excluded_from_metrics':True,'samples':records}
    (directory/'trajectory_diagnostics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    page=['<!doctype html><meta charset="utf-8"><h1>CPU-only trajectory autopsy</h1>']
    for row in records:
        name=f"{row['split']}-step-{row['step']}"
        page.append(f'<h2>{name}</h2><img style="max-width:100%" src="{name}-comparison.png"><img src="{name}-pen-histogram.png">')
    (directory/'autopsy-index.html').write_text('\n'.join(page))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    parser.add_argument('--data-root',default='data/diffink/iam_overfit')
    args=parser.parse_args()
    print(json.dumps(audit_existing(args.directory,args.data_root),indent=2))
if __name__=='__main__':main()
