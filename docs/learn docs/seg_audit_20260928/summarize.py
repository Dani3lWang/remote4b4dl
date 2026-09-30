"""Generate local, reproducible tables from the three read-only snapshots."""
import csv
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
servers = {p.stem.removeprefix('server_'): json.loads(p.read_text(encoding='utf-8-sig')) for p in BASE.glob('server_*.json')}
files = servers['21358']['files']
replay = json.loads((BASE/'mask_replay_21358.json').read_text(encoding='utf-8-sig'))

def table(headers, rows):
    def cell(x):
        if isinstance(x,float): return f'{x:.6f}'
        return str(x).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'] + ['| '+' | '.join(map(cell,r))+' |' for r in rows])

index = []
for server,d in servers.items():
    for p,e in d['files'].items():
        index.append({'server_port':server,'path':p,'bytes':e['bytes'],'mtime':e['mtime'],'sha256':e['sha256'],'kind':('json' if 'json' in e else 'prediction_index' if 'prediction_summary' in e else 'log' if 'tail' in e else 'script')})
with (BASE/'evidence_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)

parts=['# 三机 seg 实验数据附录（2026-09-28）','本文件由 summarize.py 从只读证据快照生成；原始路径与 SHA256 见 evidence_index.csv。epoch 均为日志中的零基编号。数值使用 0–1 比例。不同表代表不同评估任务，不跨表排名。']
parts += ['## 服务器证据覆盖',table(['端口','主机','代码 HEAD','采集时刻','文件数','解析错误'],[(s,d['host'],d['git_head'][:7],d['captured_at'],len(d['files']),len(d['errors'])) for s,d in servers.items()])]
parts += ['## 数据清单（现存文件，不追溯为历史运行时的文件哈希）',table(['文件','记录','场景','负例','目标','前400条独立帧'],[(pathlib.Path(p).name,m['records'],m['scenes'],m['negatives'],m['targets'],m['first400_unique_frames']) for p,m in servers['21358']['manifests'].items()])]
states=[]
for p,e in files.items():
    if p.endswith('trainer_state.json') and '/checkpoint-best/' in p and '/_protected' not in p:
        j=e['json'];states.append((pathlib.Path(p).parents[1].name,j.get('epoch'),j.get('global_step'),j.get('best_validation_iou'),p))
parts += ['## 完整模型 best checkpoint 元数据（历史 TF 口径）',table(['实验','best epoch','step','best TF mean IoU','证据'],states)]
rows=[]
for name,r in replay['runs'].items():
    m=r['metrics'];rows.append((name,r['samples'],r['tp'],r['fp'],r['fn'],m['instance_recall@0.5'],m['cIoU'],m['gIoU'],m['token_failure_rate']))
parts += ['## 自由生成掩码本次 CPU 复算', '相同样本身份与 GT 哈希已核对；6 组各 300 条，共 1,800 条，均有 330 个 GT，全部无负例。类别无关的 IoU 匹配；class_macro_f1 不作为完整识别成绩。',table(['实验','样本','TP','FP','FN','Recall@0.5','cIoU面积加权','gIoU逐对均值','token失败'],rows)]
with (BASE/'free_generation_replayed.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['run','samples','TP','FP','FN','recall','cIoU','gIoU','token_failure']);w.writerows(rows)

probe_rows=[]
for p,e in files.items():
    if p.endswith('/report.json') and any(t in p for t in ['/_oracle','/_multitask','/_temporal']):
        j=e['json'];t=j.get('test',{});probe_rows.append((pathlib.Path(p).parent.name,j.get('config',{}).get('epochs'),j.get('selected_epoch','末轮'),j.get('config',{}).get('eval_records'),t.get('objects'),t.get('recall_at_0.5'),t.get('recall_top_k','未记录'),t.get('iou_mean'),t.get('auc_mean'),t.get('predicted_size_median'),t.get('target_size_median')))
parts += ['## Oracle / 多任务 / 时序探针（非自由生成）',table(['实验','轮数','选定epoch','评估记录','目标','R@0.5','topK命中','mean IoU','AUC','预测点中位','GT点中位'],probe_rows)]
ground=[]
for p,e in files.items():
    if p.endswith('/report.json') and '/_grounding' in p:
        j=e['json']['overall'];ground.append((pathlib.Path(p).parent.name,'in-range修复' if '_inrange' in p else '旧前缀口径',j.get('n'),j.get('auc_mean'),j.get('recall_thresholded'),j.get('recall_top_k'),j.get('predicted_size_median'),j.get('target_size_median')))
parts += ['## 全部接地诊断（新旧口径分别标记）',table(['实验','口径','物体','AUC','命中率','topK命中','预测点中位','GT点中位'],ground)]
sem=[]
for p,e in files.items():
    if '/linear_probe_' in p and p.endswith('.json'):
        j=e['json'];sem.append((pathlib.Path(p).stem,j.get('final',{}).get('miou'),j.get('final',{}).get('point_accuracy'),j.get('train_samples'),j.get('val_samples')))
parts += ['## 语义线性探针（固定编码器，分类头训练2轮）','多任务实例训练场景与语义 val 存在 49 个场景重叠，不能把本表提升视为严格未见场景泛化。',table(['档位','mIoU','point acc','train帧','val帧'],sem)]

tf=[]
for p,e in files.items():
    vals=list(e.get('metric_records',[]))
    for line in e.get('selected_lines',[])+e.get('tail','').splitlines():
        if 'teacher_forcing_mean_iou' not in line:continue
        try:
            j,_=json.JSONDecoder().raw_decode(line[line.index('{'):])
            if j not in vals:vals.append(j)
        except Exception:pass
    for j in vals:
        if 'teacher_forcing_mean_iou' in j:
            tf.append({'source':p,'epoch':j.get('epoch',''),'samples':j.get('requested_samples',''),'mean_iou':j['teacher_forcing_mean_iou'],'global_iou':j.get('teacher_forcing_global_iou',''),'objects':j.get('teacher_forcing_objects',''),'recall':j.get('teacher_forcing_recall_at_0.5',''),'hits':j.get('teacher_forcing_hits_at_0.5','')})
with (BASE/'tf_all_observations.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(tf[0]));w.writeheader();w.writerows(tf)
parts += ['## 训练曲线与 TF 复核索引','逐轮数据见 tf_all_observations.csv。同一结果可能被 driver/monitor 转录，按 source 保留，不能把 CSV 行数当独立实验次数。']
for server,d in servers.items():
    logs=[]
    for p,e in d['files'].items():
        if e.get('metric_records') and p.endswith('.log'):
            ms=[r for r in e['metric_records'] if 'teacher_forcing_mean_iou' in r]
            if ms:
                best=max(ms,key=lambda r:r['teacher_forcing_mean_iou']);logs.append((p,len(ms),best.get('epoch',''),best['teacher_forcing_mean_iou'],ms[-1].get('epoch','')))
    if logs:parts += [f'### 端口 {server}',table(['日志','TF记录条数','最高值epoch','最高mean IoU','最后epoch'],logs)]

condition=[]
for p,e in files.items():
    if '/reasonseg_data_trainval/probe_' in p and '/backup' not in p and p.endswith('.json'):
        for v in e['json'].get('variants',[]):
            condition.append((pathlib.Path(p).name,v.get('variant'),v.get('iou_vs_frame_gt'),v.get('iou_vs_reference'),v.get('cosine_query_feature'),v.get('mask_area_at_threshold')))
parts += ['## 早期条件化 / marker 探针（个案机制诊断）',table(['报告','变体','GT IoU','与原掩码IoU','query余弦','预测点数'],condition)]
parts += ['## 相同文件去重','三机中相同路径与 SHA256 的文件只视为同一产物副本。预测文本相同不代表掩码相同，应同时检查 masks 的哈希。']
dup=[]
for s,d in servers.items():
    if s=='21358':continue
    for p,e in d['files'].items():
        if p in files and e['sha256']==files[p]['sha256']:dup.append((s,p,e['sha256']))
parts.append(table(['端口（与21358重复）','路径','SHA256'],dup))
(BASE/'实验数据附录.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans']
    fig,axs=plt.subplots(1,3,figsize=(16,5.8),layout='constrained')
    names=['reasonseg-trainval4090-encfp32-thr0.5-n300','reasonseg-tvenc_encfp32','reasonseg-tvenc680_encfp32','reasonseg-internal679_encfp32','reasonseg-a2ext20_encfp32']
    axs[0].barh(['旧编码器','新编码器 tvenc','tvenc680','internal679','Tversky 20ep'],[replay['runs'][n]['metrics']['instance_recall@0.5']*100 for n in names],color='#3f806f')
    axs[0].set_title('自由生成：同300条 / 330目标\n本次存档掩码复算');axs[0].set_xlabel('IoU≥0.5 命中率（%）');axs[0].invert_yaxis()
    prs=[r for r in probe_rows if r[0]!='_oracle_probe']
    labels={'_oracle_probe_v2':'冻结 v2','_oracle_probe_v3_unfrozen':'解冻 v3','_oracle_probe_v4_unfrozen_bnfix':'纯实例 v4','_multitask_encoder':'多任务20ep','_multitask_encoder_ext':'多任务30ep','_temporal_a2_repeat_f3':'复制3帧 A2'}
    axs[1].barh([labels.get(r[0],r[0]) for r in prs],[r[5]*100 for r in prs],color='#47769b')
    axs[1].set_title('Oracle：GT中心＋类别\n400条 / 416目标，非端到端');axs[1].set_xlabel('IoU≥0.5 命中率（%）');axs[1].invert_yaxis()
    keys=['linear_probe_pretrained','linear_probe_v3_unfrozen','linear_probe_v4_unfrozen_bnfix','linear_probe_mt_encoder_selected','linear_probe_ext_encoder_selected']
    lookup={r[0]:r[1] for r in sem}
    axs[2].barh(['预训练','v3','v4','多任务20ep','多任务30ep'],[lookup[k]*100 for k in keys],color='#a0794c')
    axs[2].set_title('语义线性探针：固定编码器\n存在跨任务训练/验证场景重叠');axs[2].set_xlabel('mIoU（%）');axs[2].invert_yaxis()
    for ax in axs:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('B4DL ReasonSeg 历史实验｜三类评测分别解读，不跨面板排名',fontsize=16)
    fig.savefig(BASE/'实验结果对照.png',dpi=160)
    plt.close(fig)
except ImportError as e:
    print('Plot unavailable:',e)
print(json.dumps({'evidence_files':len(index),'tf_observations':len(tf),'oracle_runs':len(probe_rows),'grounding_reports':len(ground),'linear_probes':len(sem),'duplicate_files':len(dup)}))
