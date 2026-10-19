"""Render both ALE versions from saved curves without model refitting; CHONG LIU."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/revision2'
summary=json.loads((OUT/'selection_results.json').read_text('utf-8'))
protocol=json.loads((OUT/'selection_protocol.json').read_text('utf-8'))
curves=pd.read_csv(OUT/'ALE_curves.csv')
dev=pd.read_csv(ROOT/'results/data/analysis_records.csv')
dev=dev[dev.split.isin(['train','validation'])]
sets={'validation_predefined':summary['chosen_ALE_features'],'posthoc_evaluation_selected':protocol['posthoc_ALE_features']}
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.labelsize':13,'axes.labelweight':'bold','xtick.labelsize':11.5,'ytick.labelsize':11.5,'axes.linewidth':1.5,'xtick.major.size':6,'ytick.major.size':6,'xtick.major.width':1.4,'ytick.major.width':1.4,'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False})
labels={'biogas_roll7':r'7-record mean (mL d$^{-1}$)','biogas_roll3':r'3-record mean (mL d$^{-1}$)','biogas_lag2':r'Lag 2 (mL d$^{-1}$)','daily_mean_air_temp_lag1':'Mean air temperature (°C)','daily_vpd_lag1':'Vapour-pressure deficit (kPa)'}
def draw(ax,version,f,letter):
    c=curves[(curves.version==version)&(curves.feature==f)&(curves.bins_requested==10)]
    s=dev[f].dropna();ax.axhline(0,color='#2B2B2B',lw=1,ls='--',zorder=0)
    ax.plot(c.x,c.ale_mL_d,color='#8491B4',marker='o',ms=5,lw=2.2)
    ax.plot(s,np.full(len(s),.025),'|',color='#4DBBD5',ms=5,alpha=.45,transform=ax.get_xaxis_transform())
    ax.set_xlabel(labels[f],labelpad=7);ax.set_ylabel(r'ALE (mL d$^{-1}$)',labelpad=7)
    ax.spines[['top','right']].set_visible(False);ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontweight('bold')
    ax.text(-.14,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=16)
    ax.margins(x=.03,y=.12)
def export(fig,folder,stem):
    fig.savefig(folder/(stem+'.png'),dpi=600,bbox_inches='tight',facecolor='white',metadata={'Author':'CHONG LIU'})
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight',facecolor='white',metadata={'Author':'CHONG LIU','Creator':'CHONG LIU'})
for version,fs in sets.items():
    stem='Fig09_ALE_validation_predefined' if version=='validation_predefined' else 'FigS1_ALE_posthoc'
    folder=OUT/'figures'/stem;sub=folder/'subfigures';sub.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,f in enumerate(fs):draw(axes.flat[i],version,f,chr(97+i))
    fig.subplots_adjust(left=.095,right=.985,bottom=.10,top=.95,hspace=.42,wspace=.32)
    export(fig,folder,stem);plt.close(fig)
    for i,f in enumerate(fs):
        fig,ax=plt.subplots(figsize=(6,4));draw(ax,version,f,chr(97+i));fig.tight_layout(pad=1.8)
        export(fig,sub,stem+'_'+chr(97+i));plt.close(fig)
    (sub/'README.md').write_text('Each panel is independently rendered on its own canvas from the same analysis table; panels are not crops. Author: CHONG LIU.\n',encoding='utf-8')
print('Both ALE composites and eight independent panels rendered.')
