"""End-to-end preprocessing, modelling and PDM population proof of concept."""
import json, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np, wfdb, mne
from psycopg2.extras import execute_values
from .model import AttentionAutoencoder
from .db import connect
from .evidence import compute_channel_evidence
from scripts.prepare_sleep_edf import (
    channel_sampling_rates,
    load_recording_metadata,
    meaningful_recording_rate,
    timestamp_for_recording,
    write_recording_metadata,
)

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; AP=ROOT/'data'/'apnea'; SL=ROOT/'data'/'sleep_edf'
AP_RECORDS=['a01','a02','a03','c01','c02','c03']
SL_PAIRS=[('SC4001E0','SC4001EC'),('SC4002E0','SC4002EC'),('SC4011E0','SC4011EH'),('SC4012E0','SC4012EC'),('SC4021E0','SC4021EH'),('SC4022E0','SC4022EJ')]
BASE=datetime(2000,1,1,tzinfo=timezone.utc)
SLEEP_MODELED_CHANNELS=['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal']
SLEEP_AUX_CHANNELS=['Resp oro-nasal','EMG submental','Temp rectal']
_SLEEP_RECORDING_INFO={}
_SLEEP_NORMALIZATION=None

def apnea_features(y):
    b=y.reshape(100,60); mean=b.mean(1); sd=b.std(1); lo=b.min(1); hi=b.max(1)
    slope=(b[:,-1]-b[:,0])/59.; energy=np.mean(b*b,1)
    return np.stack([mean,sd,lo,hi,hi-lo,slope,energy],1).astype('float32')

def preprocess_apnea():
    xs=[]; meta=[]; raw=[]
    for rid in AP_RECORDS:
        rec=wfdb.rdrecord(str(AP/rid)); ann=wfdb.rdann(str(AP/rid),'apn')
        for i,(sample,label) in enumerate(zip(ann.sample,ann.symbol)):
            sample=int(sample); y=rec.p_signal[sample:sample+6000,0]
            if len(y)!=6000: continue
            xs.append(apnea_features(y)); meta.append((rid,i,sample,sample+6000,label)); raw.append(y)
    x=np.asarray(xs); normal=np.array([i for i,m in enumerate(meta) if m[4]=='N'])
    mu=x[normal].mean((0,1),keepdims=True); sd=x[normal].std((0,1),keepdims=True)+1e-6; xn=(x-mu)/sd
    np.savez_compressed(OUT/'apnea_processed.npz',x=xn,raw=np.asarray(raw),meta=np.asarray(meta,dtype='U'))
    return xn,meta,np.asarray(raw)[:,:,None]

def _stage_name(s): return s.replace('Sleep stage ','').strip()


def _denormalize_sleep_reconstruction(reconstruction, normalization):
    """Convert a normalized reconstruction back to physical Sleep units."""

    if normalization is None:
        raise ValueError('Sleep physical reconstruction requires held-out-fold normalization')
    reconstruction = np.asarray(reconstruction, dtype='float32')
    mean = np.asarray(normalization['mean'], dtype='float32').reshape(1, 1, -1)
    std = np.asarray(normalization['std'], dtype='float32').reshape(1, 1, -1)
    if reconstruction.ndim != 3 or reconstruction.shape[-1] != mean.shape[-1]:
        raise ValueError('normalization parameters do not match Sleep reconstruction channels')
    return reconstruction * std + mean


def preprocess_sleep():
    global _SLEEP_NORMALIZATION
    xs=[]; meta=[]; extras=[]
    _SLEEP_RECORDING_INFO.clear()
    for pair_index, (rid,hid) in enumerate(SL_PAIRS):
        raw=mne.io.read_raw_edf(SL/f'{rid}-PSG.edf',preload=True,verbose=False); ann=mne.read_annotations(SL/f'{hid}-Hypnogram.edf')
        rates=channel_sampling_rates(raw)
        modeled_rates=[rates[n] for n in SLEEP_MODELED_CHANNELS if n in rates]
        if len(modeled_rates) != len(SLEEP_MODELED_CHANNELS):
            missing=[n for n in SLEEP_MODELED_CHANNELS if n not in rates]
            raise ValueError(f'{rid} is missing modeled Sleep channels: {missing}')
        if not all(abs(rate-modeled_rates[0]) <= 1e-9 for rate in modeled_rates[1:]):
            raise ValueError(f'{rid} modeled Sleep channels do not share a sampling rate: {modeled_rates}')
        picks=[raw.ch_names.index(n) for n in SLEEP_MODELED_CHANNELS]
        raw_rate=float(raw.info['sfreq'])
        modeled_rate=float(modeled_rates[0])
        # Keep the Phase 3 ten-Hz model representation, deriving the stride
        # from EDF metadata rather than assuming a 100-Hz source.
        model_output_rate=modeled_rate/10.0
        modeled_stride=max(1, int(round(raw_rate/model_output_rate)))
        ep=int(round(raw_rate*30.0))
        measurement_time,timestamp_provenance=timestamp_for_recording(raw, pair_index)
        _SLEEP_RECORDING_INFO[rid]={
            'channel_sampling_rates': rates,
            'recording_sampling_rate_hz': meaningful_recording_rate(rates),
            'raw_sampling_rate_hz': raw_rate,
            'modeled_sampling_rates': {n: model_output_rate for n in SLEEP_MODELED_CHANNELS},
            'aux_sampling_rates': {n: rates[n] for n in SLEEP_AUX_CHANNELS if n in rates},
            'recording_start_time': measurement_time,
            'timestamp_provenance': timestamp_provenance,
            'duration_seconds': float(raw.n_times/raw_rate),
        }
        stages=[]
        for desc,dur,on in zip(ann.description,ann.duration,ann.onset):
            st=_stage_name(str(desc)); stages += [st]*int(round(float(dur)/30))
        n=min(len(stages),raw.n_times//ep)
        xpicks=[raw.ch_names.index(n) for n in SLEEP_AUX_CHANNELS if n in raw.ch_names]
        aux_strides=[max(1,int(round(raw_rate/rates[n]))) for n in SLEEP_AUX_CHANNELS if n in rates]
        for i in range(n):
            sig=raw.get_data(picks=picks,start=i*ep,stop=(i+1)*ep).T[::modeled_stride]
            prev=stages[i-1] if i else None; transition=bool(prev is not None and stages[i]!=prev)
            aux=raw.get_data(picks=xpicks,start=i*ep,stop=(i+1)*ep).T
            if xpicks:
                aux=np.column_stack([column[::stride] for column,stride in zip(aux.T,aux_strides)])
            else:
                aux=np.empty((len(sig),0),dtype='float32')
            xs.append(sig.astype('float32')); extras.append(aux.astype('float32')); meta.append((rid,i,int(i*ep),int((i+1)*ep),stages[i],int(transition)))
    write_recording_metadata([
        dict(info, recording_id=rid)
        for rid, info in sorted(_SLEEP_RECORDING_INFO.items())
    ])
    x=np.asarray(xs); normal=np.array([i for i,m in enumerate(meta) if m[4]=='2' and not m[5]])
    mu=x[normal].mean((0,1),keepdims=True); sd=x[normal].std((0,1),keepdims=True)+1e-6; xn=(x-mu)/sd
    _SLEEP_NORMALIZATION={'mean':mu.reshape(-1), 'std':sd.reshape(-1), 'fit_scope':'normal Sleep windows in this preprocessing run'}
    np.savez_compressed(OUT/'sleep_processed.npz',x=xn,raw=x,aux=np.asarray(extras),meta=np.asarray(meta,dtype='U'),normalization_mean=mu.reshape(-1),normalization_std=sd.reshape(-1))
    return xn,meta,x,np.asarray(extras)

def _insert(cur,sql,args): cur.execute(sql,args); return cur.fetchone()[0] if cur.description else None


def _sleep_recording_info(rid):
    """Return in-memory Sleep metadata, falling back to the stable sidecar."""

    info=dict(_SLEEP_RECORDING_INFO.get(rid,{}))
    if info:
        return info
    row=load_recording_metadata().get(rid)
    if not row:
        return {}
    info.update(row)
    if 'channel_sampling_rates' not in info:
        info['channel_sampling_rates']=info.get('sampling_rates',{})
    rates=info.get('channel_sampling_rates',{})
    if info.get('raw_sampling_rate_hz') is None and rates:
        info['raw_sampling_rate_hz']=max(float(rate) for rate in rates.values())
    if not info.get('modeled_sampling_rates'):
        info['modeled_sampling_rates']={
            channel: float(rates[channel])/10.0
            for channel in SLEEP_MODELED_CHANNELS
            if channel in rates
        }
    if not info.get('aux_sampling_rates'):
        info['aux_sampling_rates']={
            channel: float(rates[channel])
            for channel in SLEEP_AUX_CHANNELS
            if channel in rates
        }
    if 'recording_start_time' in info and isinstance(info['recording_start_time'],str):
        info['recording_start_time']=datetime.fromisoformat(info['recording_start_time'])
    return info


def populate(kind,x,meta,physical,aux,model,res,normalization=None,channel_thresholds=None):
    if kind == 'sleep':
        if normalization is None:
            raise ValueError('Sleep population requires held-out-fold normalization parameters')
        if channel_thresholds is None:
            raise ValueError('Sleep population requires validation-derived channel thresholds')
        channel_thresholds=np.asarray(channel_thresholds,dtype='float64')
        reconstructed_physical=_denormalize_sleep_reconstruction(res['reconstruction'],normalization)
    else:
        reconstructed_physical=None
    # keep at most 60 minutes/recording (all processed windows are used for outputs)
    with connect() as conn, conn.cursor() as c:
        model_id=_insert(c,"INSERT INTO model(model_name,model_version,task,architecture,training_dataset,input_configuration) VALUES(%s,'poc-1','anomaly detection','BiLSTM + learned temporal attention autoencoder',%s,%s) ON CONFLICT(model_name,model_version) DO UPDATE SET architecture=EXCLUDED.architecture,task=EXCLUDED.task RETURNING model_id",(f'{kind}_attention_autoencoder',kind,str(x.shape[1:])))
        if model_id is None: c.execute("SELECT model_id FROM model WHERE model_name=%s AND model_version='poc-1'",(f'{kind}_attention_autoencoder',)); model_id=c.fetchone()[0]
        c.execute("INSERT INTO model_modality(model_id,modality) VALUES(%s,%s) ON CONFLICT DO NOTHING",(model_id,kind))
        for rid in sorted(set(m[0] for m in meta)):
            code=('APNEA_'+rid) if kind=='apnea' else ('SLEEP_'+rid[:5])
            c.execute("INSERT INTO patient(patient_code) VALUES(%s) ON CONFLICT(patient_code) DO UPDATE SET patient_code=EXCLUDED.patient_code RETURNING patient_id",(code,)); pid=c.fetchone()[0]
            info=_sleep_recording_info(rid) if kind=='sleep' else {}
            recording_rate=info.get('recording_sampling_rate_hz') if kind=='sleep' else 100.0
            recording_start=info.get('recording_start_time') if kind=='sleep' else BASE
            recording_length=info.get('duration_seconds',0.0) if kind=='sleep' else 0.0
            c.execute("INSERT INTO recording(recording_id,patient_id,source_recording_name,modality,sampling_rate_hz,length_seconds,recording_start_time) VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(recording_id) DO UPDATE SET patient_id=EXCLUDED.patient_id, sampling_rate_hz=EXCLUDED.sampling_rate_hz, length_seconds=EXCLUDED.length_seconds, recording_start_time=EXCLUDED.recording_start_time RETURNING recording_id",(f'{kind}_{rid}',pid,rid,kind,recording_rate,recording_length,recording_start)); recid=c.fetchone()[0]
            channels=['ECG'] if kind=='apnea' else SLEEP_MODELED_CHANNELS+SLEEP_AUX_CHANNELS
            modeled_count=physical.shape[2]
            modeled_rates=([100.0]*modeled_count if kind=='apnea' else [info.get('modeled_sampling_rates',{}).get(ch) for ch in SLEEP_MODELED_CHANNELS[:modeled_count]])
            aux_rates=([1.0]*max(0,len(channels)-modeled_count) if kind=='apnea' else [info.get('aux_sampling_rates',{}).get(ch) for ch in SLEEP_AUX_CHANNELS])
            sids=[]; recon_sids=[]
            for ch in channels:
                rate=modeled_rates[channels.index(ch)] if ch in channels[:modeled_count] else aux_rates[channels.index(ch)-modeled_count]
                c.execute("INSERT INTO signal(recording_id,signal_type,channel,sampling_rate_hz) VALUES(%s,%s,%s,%s) RETURNING signal_id",(recid,kind,ch,rate)); sids.append(c.fetchone()[0])
            for ci,ch in enumerate(channels[:modeled_count]):
                c.execute("INSERT INTO signal(recording_id,signal_type,channel,sampling_rate_hz) VALUES(%s,%s,%s,%s) RETURNING signal_id",(recid,kind+'_reconstruction',ch,modeled_rates[ci])); recon_sids.append(c.fetchone()[0])
            candidates=[j for j,m in enumerate(meta) if m[0]==rid]
            event=next((j for j in candidates if (m:=meta[j]) and (m[4]=='A' if kind=='apnea' else int(m[5])==1)),candidates[0])
            radius=30 if kind=='apnea' else 60
            selected=set(candidates[:min(10,len(candidates))]) | set(range(max(candidates[0],event-radius),min(candidates[-1]+1,event+radius)))
            for j in sorted(selected):
                m=meta[j]
                if kind == 'sleep':
                    raw_rate=float(info['raw_sampling_rate_hz'])
                    st=info['recording_start_time']+timedelta(seconds=float(m[2])/raw_rate)
                else:
                    st=BASE+timedelta(seconds=float(m[2])/100.0)
                en=st+timedelta(seconds=60 if kind=='apnea' else 30)
                c.execute("INSERT INTO analysis_window(recording_id,window_index,start_sample,end_sample,start_time,end_time,duration_seconds) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING window_id",(recid,int(m[1]),int(m[2]),int(m[3]),st,en,(en-st).total_seconds())); wid=c.fetchone()[0]
                for sid in sids+recon_sids: c.execute("INSERT INTO window_signal VALUES(%s,%s)",(wid,sid))
                label=m[4]
                c.execute("INSERT INTO annotation(recording_id,window_id,annotation_type,label,start_time,end_time,source) VALUES(%s,%s,%s,%s,%s,%s,%s)",(recid,wid,'sleep_stage' if kind=='sleep' else 'apnea',str(label),st,en,'official'))
                if kind=='sleep' and int(m[5]):
                    c.execute("INSERT INTO annotation(recording_id,window_id,annotation_type,label,start_time,end_time,source) VALUES(%s,%s,'stage_transition','transition',%s,%s,%s)",(recid,wid,st,en,'derived_prototype'))
                is_anomaly = bool(res['is_anomaly'][j])
                predicted_class = 'anomaly' if is_anomaly else 'normal'
                c.execute("INSERT INTO prediction(window_id,model_id,predicted_class,reconstruction_error,anomaly_threshold,is_anomaly,severity,status) VALUES(%s,%s,%s,%s,%s,%s,%s,'complete') RETURNING prediction_id",(wid,model_id,predicted_class,float(res['reconstruction_error'][j]),float(model.threshold),is_anomaly,'high' if is_anomaly else 'normal')); pred=c.fetchone()[0]
                c.execute("INSERT INTO xai_explanation(prediction_id,xai_method,explanation_type,explanation_text) VALUES(%s,'temporal_attention','window_level_temporal_relevance',%s) RETURNING explanation_id",(pred,'Shared window-level temporal relevance; no channel owner')); expl=c.fetchone()[0]
                execute_values(c,"INSERT INTO attention_data(timestamp,window_id,signal_id,attention_value) VALUES %s",[(st+timedelta(seconds=float(k)*(60 if kind=='apnea' else 30)/len(res['attention'][j])),wid,None,float(v)) for k,v in enumerate(res['attention'][j])])
                for ci,sid in enumerate(sids[:modeled_count]):
                    hz=float(modeled_rates[ci])
                    execute_values(c,"INSERT INTO signal_data(timestamp,signal_id,channel,value) VALUES %s",[(st+timedelta(seconds=k/hz),sid,channels[ci],float(v)) for k,v in enumerate(physical[j,:,ci])])
                    rsid=recon_sids[ci]
                    if kind != 'apnea':
                        execute_values(c,"INSERT INTO signal_data(timestamp,signal_id,channel,value) VALUES %s",[(st+timedelta(seconds=k/hz),rsid,channels[ci],float(v)) for k,v in enumerate(reconstructed_physical[j,:,ci])])
                if kind == 'sleep':
                    evidence_times=np.asarray([st+timedelta(seconds=k/float(modeled_rates[0])) for k in range(x.shape[1])],dtype=object)
                    evidence=compute_channel_evidence(x[j],res['reconstruction'][j],channel_thresholds,evidence_times)
                    evidence_rows=[]
                    for ci,row in enumerate(evidence):
                        rsid=sids[ci]
                        c.execute("INSERT INTO xai_location(explanation_id,signal_id,channel,temporal_start,temporal_end,spatial_location) VALUES(%s,%s,%s,%s,%s,%s)",(expl,rsid,channels[ci],row['peak_residual_time'],row['peak_residual_time'],'peak_residual'))
                        for name in ('mean_error','max_error','contribution_share','validation_threshold','raw_margin','normalized_margin','residual_threshold_flag'):
                            value=row[name]
                            evidence_rows.append((expl,rsid,name,float(value) if not isinstance(value,bool) else float(value),None,ci+1))
                    execute_values(c,"INSERT INTO xai_feature(explanation_id,signal_id,feature_name,feature_value,attribution_value,importance_rank) VALUES %s",evidence_rows)
                if kind == 'apnea':
                    execute_values(c,"INSERT INTO signal_feature_data(timestamp,window_id,signal_id,feature_name,feature_value) VALUES %s",[(st+timedelta(seconds=k*60.0/res['reconstruction'].shape[1]),wid,recon_sids[0],name,float(res['reconstruction'][j,k,fi])) for k in range(res['reconstruction'].shape[1]) for fi,name in enumerate(['mean','std','min','max','range','slope','energy'])],page_size=10000)
                if aux is not None:
                    for ci,sid in enumerate(sids[3:]):
                        hz=float(aux_rates[ci])
                        execute_values(c,"INSERT INTO signal_data(timestamp,signal_id,channel,value) VALUES %s",[(st+timedelta(seconds=k/hz),sid,channels[ci+modeled_count],float(v)) for k,v in enumerate(aux[j,:,ci])])
        conn.commit()

def run():
    """Refuse the pre-Phase-6 training/reset workflow.

    Database population is owned by ``scripts.populate_fusion_pdm`` and its
    validated artifact contract; this legacy entry point must remain a hard
    safety boundary so ordinary execution cannot retrain or rewrite PDM rows.
    """

    raise RuntimeError(
        'pdm.pipeline.run() is disabled: use validated LOSO artifacts from '
        'Phase 3 '
        'and the Phase 6 population workflow; global training and table resets '
        'are not an evidence-producing path'
    )
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == '__main__':
    main()
