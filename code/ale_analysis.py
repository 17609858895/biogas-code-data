from pathlib import Path
import os,sys,json,hashlib,warnings
warnings.filterwarnings('ignore',message='X does not have valid feature names')
os.environ['OMP_NUM_THREADS']='2'
import numpy as np,pandas as pd,joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
_original_savefig=Figure.savefig
def _author_savefig(self,*args,**kwargs):
    kwargs.setdefault('metadata',{}).update({'Author':'CHONG LIU'})
    if str(args[0]).lower().endswith('.pdf'):kwargs['metadata']['Creator']='CHONG LIU'
    return _original_savefig(self,*args,**kwargs)
Figure.savefig=_author_savefig
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'results'; OUT=DATA/'figures/Fig09_ALE_reproduction';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'code'))
from ml_publication_style import configure_publication_style,style_axis,numeric_ticks,save_figure_with_subfigures
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
modelpath=DATA/'models/final_pipeline.joblib';pipe=joblib.load(modelpath)
protocol=json.loads((DATA/'audit/protocol.json').read_text('utf-8'))
d=pd.read_csv(DATA/'data/analysis_records.csv');x=d.loc[d.split.isin(['train','validation']),protocol['features']].copy()
# Adapted from the installed sac_adsorption_energy_templates. Disjoint bins
# prevent duplicate counting; empirical centering uses interpolated effects.
def ale(model,x,f,bins=10):
 z=x.loc[x[f].notna()].copy();s=z[f].to_numpy(float)
 edges=np.unique(np.quantile(s,np.linspace(0,1,bins+1)))
 ids=np.clip(np.searchsorted(edges,s,side='left')-1,0,len(edges)-2)
 changes=[];counts=[]
 for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
  low=z.loc[ids==k].copy();high=low.copy();counts.append(len(low))
  assert len(low)>0
  low[f]=lo;high[f]=hi
  changes.append(float(np.mean(model.predict(high)-model.predict(low))))
 raw=np.r_[0,np.cumsum(changes)];center=float(np.mean(np.interp(s,edges,raw)))
 curve=raw-center
 assert abs(np.mean(np.interp(s,edges,curve)))<1e-8
 assert sum(counts)==len(s)
 return pd.DataFrame({'feature':f,'bins_requested':bins,'x':edges,'ale_mL_d':curve}),pd.DataFrame({'feature':f,'bins_requested':bins,'lower':edges[:-1],'upper':edges[1:],'n':counts,'mean_local_difference_mL_d':changes})
# Validate the estimator against a known additive model on correlated data.
class Linear:
 def predict(self,z):return 3*z.a-2*z.b+7
t=pd.DataFrame({'a':np.arange(51,dtype=float),'b':np.arange(51,dtype=float)*.8})
test,_=ale(Linear(),t,'a',10);assert np.allclose(test.ale_mL_d,3*(test.x-t.a.mean()))
features=['biogas_roll7','biogas_roll3','daily_mean_air_temp_lag1','biogas_lag2']
labels=[r'7-record mean (mL d$^{-1}$)',r'3-record mean (mL d$^{-1}$)',r'Mean air temperature (°C)',r'Lag 2 (mL d$^{-1}$)']
curves=[];binsdata=[];summ=[]
configure_publication_style();fig,axes=plt.subplots(2,2,figsize=(12,8))
for k,(f,label,ax) in enumerate(zip(features,labels,axes.flat)):
 allc={}
 for b in [8,10,15]:
  c,n=ale(pipe,x,f,b);curves.append(c);binsdata.append(n);allc[b]=c
 c=allc[10];s=x[f].dropna();ax.axhline(0,color='#2B2B2B',lw=1,ls='--',zorder=0)
 ax.plot(c.x,c.ale_mL_d,color='#8491B4',marker='o',ms=4.5,lw=2.2)
 ax.plot(s,np.full(len(s),.025),'|',color='#4DBBD5',ms=5,alpha=.45,transform=ax.get_xaxis_transform())
 style_axis(ax,xlabel=label,ylabel=r'Accumulated local effect (mL d$^{-1}$)',tick_size=10.5,label_size=12)
 numeric_ticks(ax,n=4)
 ax.text(-.17,1.04,chr(97+k),transform=ax.transAxes,fontweight='bold',fontsize=16)
 ax.margins(x=.03,y=.12)
 for tick in ax.get_xticklabels():tick.set_rotation(0)
 summ.append({'feature':f,'n':len(s),'range_x':[float(s.min()),float(s.max())],'ale_range_10':[float(c.ale_mL_d.min()),float(c.ale_mL_d.max())],'bins':{b:{'net_change':float(v.ale_mL_d.iloc[-1]-v.ale_mL_d.iloc[0]),'range':float(v.ale_mL_d.max()-v.ale_mL_d.min())} for b,v in allc.items()}})
fig.subplots_adjust(left=.11,right=.985,bottom=.10,top=.955,hspace=.36,wspace=.34)
save_figure_with_subfigures(fig,OUT,'Fig09_ALE',dpi=600)
# Apply author metadata after the template's exports.
pd.concat(curves).to_csv(OUT/'ALE_curves.csv',index=False)
pd.concat(binsdata).to_csv(OUT/'ALE_bin_support.csv',index=False)
report={'model_sha256':sha(modelpath),'workbook_sha256':protocol['source_sha256'],'fit_or_data_changed':False,'reference':'424 development records; missing plotted features excluded separately','features_selected':'four highest-ranked continuous features in RF evaluation permutation output; post-hoc evaluation-informed analysis','centering':'mean interpolated ALE over nonmissing development values equals zero','validation':'known additive predictor slope and centering; exhaustive disjoint bins','curves':summ}
(OUT/'ALE_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
