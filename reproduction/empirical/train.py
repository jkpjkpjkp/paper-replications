"""Table-2 forecasting reconstruction with full-size networks and resumable fits."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np
import pandas as pd
from build_panel import DEFAULT, FIN, save_json


def load(out):
    x=np.load(out/'x.npy',mmap_mode='r');y=np.load(out/'y.npy',mmap_mode='r')
    meta=pd.read_parquet(out/'samples.parquet')
    target=FIN.index('ebit')
    observed=np.isfinite(y[:,-1,target])
    train=np.flatnonzero(meta.split.eq('train') & observed)
    valid=np.flatnonzero(meta.split.eq('valid') & observed)
    predict=np.flatnonzero(meta.split.isin(['test','gap']))
    return x,y,meta,target,train,valid,predict


def baseline(out):
    x,y,meta,t,tr,va,pr=load(out)
    folder=out/'baselines';folder.mkdir(exist_ok=True)
    predictions={'QFM':np.asarray(x[pr,-1,t],dtype=float)}
    for name,ar in [('AutoReg',True),('Linear',False)]:
        d=5 if ar else x.shape[1]*x.shape[2]
        xx=np.zeros((d+1,d+1));xy=np.zeros(d+1)
        for start in range(0,len(tr),10000):
            idx=tr[start:start+10000]
            a=np.asarray(x[idx,:,t] if ar else x[idx].reshape(len(idx),-1),dtype=float)
            a=np.column_stack([np.ones(len(idx)),a])
            xx+=a.T@a;xy+=a.T@np.asarray(y[idx,-1,t],dtype=float)
        beta=np.linalg.lstsq(xx,xy,rcond=1e-12)[0]
        np.save(folder/f'{name}_coefficients.npy',beta)
        pred=np.empty(len(pr))
        for start in range(0,len(pr),10000):
            idx=pr[start:start+10000]
            a=x[idx,:,t] if ar else x[idx].reshape(len(idx),-1)
            pred[start:start+len(idx)]=beta[0]+a@beta[1:]
        predictions[name]=pred
    report={}
    test=meta.iloc[pr].split.eq('test').to_numpy()&np.isfinite(y[pr,-1,t])
    truth=np.asarray(y[pr,-1,t])
    for name,pred in predictions.items():
        np.savez(folder/f'{name}_predictions.npz',indices=pr,mean=pred)
        mse=float(np.mean((truth[test]-pred[test])**2))
        report[name]={'test_standardized_ebit_mse':mse,'test_samples':int(test.sum()),
                      'training_samples':len(tr),'validation_samples':len(va)}
    save_json(folder/'metrics.json',report)
    print(json.dumps(report,indent=2),flush=True)


def network(kind,n_inputs,n_outputs):
    import tensorflow as tf
    L=tf.keras.layers
    uq=kind.startswith('uq_');rnn=kind.endswith('lstm')
    inputs=L.Input((5,n_inputs))
    if rnn:
        layer=L.LSTM(512,return_sequences=True,recurrent_dropout=.25,
            kernel_initializer='glorot_uniform',kernel_constraint=tf.keras.constraints.MaxNorm(3))
        h=layer(inputs,training=True) if uq else layer(inputs)
        h=L.BatchNormalization()(h)
    else:
        h=L.Flatten()(inputs)
        h=L.Dense(2048,use_bias=False,kernel_initializer='glorot_uniform',
            kernel_constraint=tf.keras.constraints.MaxNorm(3))(h)
        h=L.BatchNormalization()(h);h=L.ReLU()(h)
        dropout=L.Dropout(.25)
        h=dropout(h,training=True) if uq else dropout(h)
    mean=L.Dense(n_outputs,name='mean')(h)
    if uq:
        raw=L.Dense(n_outputs,name='variance_logits')(h)
        var=L.Lambda(lambda z:tf.maximum(tf.nn.softplus(z),1e-6),name='variance')(raw)
        return tf.keras.Model(inputs,[mean,var])
    return tf.keras.Model(inputs,mean)


def loss_value(truth,mean,var,rnn,target):
    import tensorflow as tf
    if not rnn: truth=truth[:,-1,:]
    mask=tf.math.is_finite(truth)
    clean=tf.where(mask,truth,tf.zeros_like(truth))
    err=tf.square(mean-clean)
    losses=err if var is None else err/var+tf.math.log(var)
    def masked(values,valid):
        return tf.math.divide_no_nan(tf.reduce_sum(tf.where(valid,values,0.)),
                                     tf.reduce_sum(tf.cast(valid,tf.float32)))
    if rnn:
        primary=masked(losses[:,-1,target],mask[:,-1,target])
        last=masked(losses[:,-1,:],mask[:,-1,:]);whole=masked(losses,mask)
        return .5*primary+.5*(.7*last+.3*whole)
    return .75*masked(losses[:,target],mask[:,target])+.25*masked(losses,mask)


def fit(out,kind,seed,epochs=150,draws=10):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu,True)
    tf.keras.utils.set_random_seed(seed)
    x,y,meta,target,tr,va,pr=load(out)
    folder=out/kind/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        print('Already completed',kind,seed,flush=True);return
    model=network(kind,x.shape[-1],y.shape[-1]);uq=kind.startswith('uq_');rnn=kind.endswith('lstm')
    opt=tf.keras.optimizers.Adadelta(learning_rate=.01,global_clipnorm=1.)
    checkpoint=tf.train.Checkpoint(model=model,optimizer=opt)
    manager=tf.train.CheckpointManager(checkpoint,str(folder/'resume'),max_to_keep=1)
    history=[];best=np.inf;start_epoch=0
    if (folder/'progress.json').exists() and manager.latest_checkpoint:
        state=json.loads((folder/'progress.json').read_text())
        checkpoint.restore(manager.latest_checkpoint)
        history=state['history'];best=state['best_validation_mse'];start_epoch=len(history)
    @tf.function(reduce_retracing=True)
    def train_step(a,b):
        with tf.GradientTape() as tape:
            output=model(a,training=True)
            mean,var=output if uq else (output,None)
            loss=loss_value(b,mean,var,rnn,target)
        gradients=tape.gradient(loss,model.trainable_variables)
        opt.apply_gradients(zip(gradients,model.trainable_variables))
        return loss
    @tf.function(reduce_retracing=True)
    def predict_step(a):
        output=model(a,training=False)
        mean,var=output if uq else (output,tf.zeros_like(output))
        if rnn:return mean[:,-1,target],var[:,-1,target]
        return mean[:,target],var[:,target]
    print('Training',kind,seed,'train',len(tr),'valid',len(va),
          'devices',tf.config.list_physical_devices('GPU'),flush=True)
    for epoch in range(start_epoch,epochs):
        if history and len(history)-1-int(np.argmin([h['validation_mse'] for h in history]))>=25:break
        begun=time.time();order=np.random.default_rng(seed+epoch).permutation(tr)
        total=0.
        for start in range(0,len(order),256):
            idx=order[start:start+256]
            value=float(train_step(np.asarray(x[idx]),np.asarray(y[idx])))
            if not np.isfinite(value):raise RuntimeError('Non-finite training loss')
            total+=value*len(idx)
        sse=0.
        for start in range(0,len(va),1024):
            idx=va[start:start+1024];pred,_=predict_step(np.asarray(x[idx]))
            sse+=float(np.sum((np.asarray(pred,dtype=float)-y[idx,-1,target])**2))
        val=sse/len(va)
        row={'epoch':epoch+1,'training_loss':total/len(tr),'validation_mse':val,'seconds':time.time()-begun}
        history.append(row)
        if val<best:
            best=val;model.save_weights(str(folder/'best.weights.h5'))
        manager.save(checkpoint_number=epoch+1)
        save_json(folder/'progress.json',{'kind':kind,'seed':seed,'best_validation_mse':best,'history':history})
        print(kind,seed,json.dumps(row),flush=True)
    model.load_weights(str(folder/'best.weights.h5'))
    means=np.zeros(len(pr));second=np.zeros(len(pr));noise=np.zeros(len(pr))
    ndraw=draws if uq else 1
    for draw in range(ndraw):
        for start in range(0,len(pr),1024):
            idx=pr[start:start+1024];mu,var=predict_step(np.asarray(x[idx]));mu=np.asarray(mu,dtype=float)
            means[start:start+len(idx)]+=mu/ndraw
            second[start:start+len(idx)]+=mu**2/ndraw
            noise[start:start+len(idx)]+=np.asarray(var,dtype=float)/ndraw
    variance=noise+np.maximum(second-means**2,0)
    np.savez(folder/'predictions.npz',indices=pr,mean=means,variance=variance)
    test=meta.iloc[pr].split.eq('test').to_numpy()&np.isfinite(y[pr,-1,target])
    mse=float(np.mean((means[test]-y[pr[test],-1,target])**2))
    save_json(folder/'complete.json',{'kind':kind,'seed':seed,'epochs':len(history),
        'best_validation_mse':best,'test_standardized_ebit_mse':mse,'mc_draws':ndraw,
        'train_samples':len(tr),'validation_samples':len(va),'test_samples':int(test.sum()),
        'gpu':str(tf.config.list_physical_devices('GPU'))})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--kind',choices=['baselines','mlp','lstm','uq_mlp','uq_lstm'],required=True)
    parser.add_argument('--seed',type=int,default=521);parser.add_argument('--epochs',type=int,default=150)
    args=parser.parse_args()
    if args.kind=='baselines':baseline(args.out)
    else:fit(args.out,args.kind,args.seed,args.epochs)
