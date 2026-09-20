"""Re-render cached analyses with the original palette; no model fitting."""
from pathlib import Path
import importlib.util, sys, inspect, json, pickle, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter, FixedLocator, FormatStrFormatter
import joblib

OUT=Path(__file__).resolve().parent.parent
BASE=OUT.parent
DATA=OUT/'results'
sys.path.insert(0,str(OUT/'code'))
from ml_publication_style import configure_publication_style
spec=importlib.util.spec_from_file_location('plots',OUT/'code/analysis_functions.py')
a=importlib.util.module_from_spec(spec);sys.modules['plots']=a;spec.loader.exec_module(a)
DATA=OUT/'results'
a.ROOT=DATA;a.TABLE_DIR=DATA/'tables';a.AUDIT_DIR=DATA/'audit'
FIG=DATA/'figures';FIG.mkdir(parents=True,exist_ok=True)
QA=DATA/'audit';QA.mkdir(parents=True,exist_ok=True)
mapping={'Fig01_locked_split': 'Fig01_locked_split', 'Fig02_association_analysis': 'Fig02_association_analysis', 'Fig03_validation_selection': 'Fig03_validation_selection', 'Fig04_test_prediction': 'Fig04_test_prediction', 'Fig05_calibration_coverage': 'Fig06_calibration_coverage', 'Fig06_residual_reliability': 'Fig05_residual_reliability', 'Fig07_frozen_model_importance': 'Fig07_frozen_model_importance', 'Fig08_sensitivity': 'Fig08_sensitivity'}
original_config=a.configure_style
def configure():
    original_config();configure_publication_style()
    a.np.random.seed(a.RANDOM_STATE)
a.configure_style=configure

replace={'Primary source-valid series':'Retained series','Primary source-valid':'Retained records','Positive source intervals':'Positive intervals','Selected ML model':'Selected RF','Selected model':'Reference model','Selected':'Reference','Biogas rate at STP':'Corrected rate','Gas temperature':'Temperature proxy','biogas rate at STP':'corrected rate','Biogas at STP':'Corrected rate','Biogas rate':'Corrected rate','Final-test':'Evaluation','Test date':'Evaluation date','Test RMSE':'Evaluation RMSE','Test MAE':'Evaluation MAE','Test R':'Evaluation R','Test MASE':'Evaluation MASE','Parsimonious history + operation':'History + operation','Permutation RMSE increase':'RMSE increase','Mean absolute SHAP value':'Mean |SHAP|','Mean observed and predicted rate':'Mean observed / predicted','Within-method contribution':'Contribution','Pearson correlation coefficients (r)':'Pearson r','Spearman correlation coefficients (rho)':'Spearman ρ','Mean |Spearman rho| with biogas':'Mean |Spearman ρ|','Mutual information with biogas':'Mutual information'}
features={'biogas_roll7':'7-record mean','biogas_roll3':'3-record mean','biogas_lag1':'Lag 1','biogas_lag2':'Lag 2','biogas_delta1':'Rate change','reactor_id':'Reactor ID','reactor_type':'Reactor type','daily_mean_air_temp_lag1':'Mean air temp.','daily_temp_range_lag1':'Temp. range','daily_solar_lag1':'Solar radiation','days_since_prev':'History age','daily_max_air_temp_lag1':'Max. air temp.','reactor_type_FLEX':'Flexible','reactor_type_FIXED DOME':'Fixed dome','reactor_id_R4-FIXED DOME':'R4 fixed dome','reactor_id_R2-FLEX':'R2 flexible','air_temp_in_situ_lag1':'Local air temp.','daily_precip_lag1':'Precipitation','daily_vpd_lag1':'VPD','manure_fed_kg':'Manure feed','water_kg':'Water feed'}
heat={'Biogas_STP':'Rate','Biogas_lag1':'Lag 1','Biogas_lag2':'Lag 2','Biogas_roll3':'Mean 3','Biogas_roll7':'Mean 7','Biogas_delta1':'Δrate','Manure':'Feed','Water':'Water','T_local':'Tloc','T_mean':'Tavg','Solar':'Solar','Precip':'Rain','VPD':'VPD','T_range':'ΔT'}
records=[]
heat.update({'Biogas_STP':'q','Biogas_lag1':'L1','Biogas_lag2':'L2','Biogas_roll3':'M3','Biogas_roll7':'M7','Biogas_delta1':'Δq'})
heat.update({'Manure':'F','Water':'W','T_local':'Tl','T_mean':'Tm','Solar':'S','Precip':'P'})
replace.update({'Historical biogas':'History','Operational conditions':'Operation','Weather context':'Weather'})
replace.update({'Selected ML model':'RF','Selected model':'RF','Selected':'RF'})
replace['Test $R^2$']='Evaluation $R^2$'

def align_outer_footprints(fig,axes):
    """Fit each panel's full decoration footprint to an aligned grid cell."""
    arr=a.np.asarray(axes)
    if arr.ndim==1:arr=arr.reshape((-1,1) if len(arr)==4 else (1,-1))
    fig.canvas.draw();ren=fig.canvas.get_renderer()
    bounds=[ax.get_tightbbox(ren).transformed(fig.transFigure.inverted()) for ax in arr.flat]
    targets=[]
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            col=[bounds[rr*arr.shape[1]+c] for rr in range(arr.shape[0])]
            row=bounds[r*arr.shape[1]:(r+1)*arr.shape[1]]
            targets.append((min(b.x0 for b in col), min(b.y0 for b in row),max(b.x1 for b in col), max(b.y1 for b in row)))
    for _ in range(3):
        fig.canvas.draw();ren=fig.canvas.get_renderer()
        for ax,t in zip(arr.flat,targets):
            if ax.get_aspect()!='auto':continue
            b=ax.get_tightbbox(ren).transformed(fig.transFigure.inverted());p=ax.get_position()
            x0=p.x0+t[0]-b.x0;y0=p.y0+t[1]-b.y0
            x1=p.x1+t[2]-b.x1;y1=p.y1+t[3]-b.y1
            if x1>x0 and y1>y0:ax.set_position([x0,y0,x1-x0,y1-y0])
    fig.canvas.draw()
    return [[round(v,5) for v in ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds] for ax in arr.flat]

def export_file(fig,path,dpi=600):
    fig.savefig(path.with_suffix('.png'),dpi=dpi,bbox_inches='tight',pad_inches=.12,facecolor='white',metadata={'Author':'CHONG LIU'})
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.12,facecolor='white',metadata={'Author':'CHONG LIU','Creator':'CHONG LIU'})

def export_subfigures(fig,axes,stem):
    """Independent vector-artist renders on newly sized canvases, never image crops."""
    folder=FIG/stem/'subfigures';folder.mkdir(parents=True,exist_ok=True)
    serialized=pickle.dumps(fig)
    main=list(a.np.asarray(axes).flat)
    indices=[fig.axes.index(ax) for ax in main]
    for n,index in enumerate(indices):
        single=pickle.loads(serialized);ax=single.axes[index]
        for other in list(single.axes):
            if other is not ax:single.delaxes(other)
        for leg in list(single.legends):leg.remove()
        # Reintroduce the shared reactor key only for panels with reactor-series artists.
        h,l=ax.get_legend_handles_labels()
        if ax.get_legend() is None and h and stem.startswith(('Fig04','Fig05')):
            ax.legend(h,l,frameon=False,loc='lower left',bbox_to_anchor=(0,1.08),ncol=2,fontsize=11)
        wide=stem.startswith('Fig06')
        single.set_size_inches((10.2,3.7) if wide else (8.4,6.5))
        if stem.startswith('Fig02') and n<2:single.set_size_inches(10.5,8.5)
        ax.set_position([.20,.18,.73,.68] if not wide else [.13,.22,.83,.53])
        if not ax.get_xlabel() and wide:ax.set_xlabel('Evaluation date',fontweight='bold')
        export_file(single,folder/f'{stem}_{chr(97+n)}')
        plt.close(single)

def save_figure_with_subfigures(fig,stem,axes):
    stem=mapping[stem];arr=list(a.np.asarray(axes).flat)
    if '--only' in sys.argv and stem[:5] not in sys.argv[sys.argv.index('--only')+1].split(','):
        plt.close(fig);return
    for t in fig.findobj(Text):
        txt=t.get_text()
        for old,new in replace.items():txt=txt.replace(old,new)
        t.set_text(txt)
    for ax in arr:
        for t in ax.get_xticklabels():t.set_rotation(0);t.set_ha('center')
    if stem.startswith('Fig02'):
        for ax in arr[:2]:
            ax.set_xticks(ax.get_xticks(),[heat.get(t.get_text(),t.get_text()) for t in ax.get_xticklabels()],rotation=0,fontsize=9.8,fontweight='bold')
            ax.set_yticks(ax.get_yticks(),[heat.get(t.get_text(),t.get_text()) for t in ax.get_yticklabels()],fontsize=11,fontweight='bold')
        ax=arr[2];ax.set_yticks(ax.get_yticks(),[heat.get(t.get_text(),t.get_text()) for t in ax.get_yticklabels()],fontweight='bold')
    if stem.startswith('Fig05'):
        for ax in arr[2:]:ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=11)
        arr[2].set_xticks(arr[2].get_xticks(),['All records','Peak records'],fontweight='bold')
        arr[3].set_xticks(arr[3].get_xticks(),['1-day history','Longer gaps'],fontweight='bold')
        arr[3].yaxis.set_major_locator(FixedLocator([140,180,220]));arr[3].yaxis.set_major_formatter(FormatStrFormatter('%g'));arr[3].yaxis.set_minor_locator(NullLocator())
        for ax in arr:
            for label in ax.texts:
                if label.get_text() in ['(a)','(b)','(c)','(d)']:label.set_position((-.13,1.05))
    if stem.startswith('Fig07'):
        fig.set_size_inches(18.2,6.7)
        for ax in arr[:2]:
            ax.set_yticks(ax.get_yticks(),[features.get(t.get_text(),t.get_text()) for t in ax.get_yticklabels()],fontweight='bold')
            ax.xaxis.set_major_locator(MaxNLocator(4))
        arr[2].legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.03),ncol=2,fontsize=11)
        arr[2].xaxis.set_major_locator(MaxNLocator(4));arr[2].set_xlim(0,100)
        for ax in arr:ax.xaxis.label.set_fontsize(14)
    if stem.startswith('Fig08'):
        for ax in arr:
            ax.set_xscale('linear');ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_yticks(ax.get_yticks(),[t.get_text().replace('Parsimonious history + operation','History + operation') for t in ax.get_yticklabels()],fontweight='bold')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');fig.tight_layout(pad=1.5,w_pad=2.4,h_pad=2.4,rect=(0,0,1,.97) if fig.legends else None)
    footprints=align_outer_footprints(fig,axes)
    export_file(fig,FIG/stem,dpi=150 if '--draft' in sys.argv else 600)
    if '--draft' not in sys.argv:export_subfigures(fig,axes,stem)
    records.append({'figure':stem,'horizontal_x_ticks':all(t.get_rotation()==0 for ax in arr for t in ax.get_xticklabels()),'outer_footprints':footprints,'panels':len(arr)})
    print(stem,flush=True);plt.close(fig)
a.save_composite_figure=save_figure_with_subfigures
source=inspect.getsource(a.create_figures)
source=source.replace('if plain == "reactor_id" or plain.startswith("reactor_id_"):','if plain in {"reactor_id", "reactor_type"} or plain.startswith(("reactor_id_", "reactor_type_")):')
source=source.replace('grouped.plot(kind="bar",','grouped.plot(kind="barh",')
source=source.replace('style_axis(ax, "Predictor group", "Within-method contribution (%)")','style_axis(ax, "Within-method contribution (%)", "")')
start=source.index('    # Figure 3:')
end=source.index('    # Figure 4:',start)
source=source[:start]+'''    # Common-evaluation comparison; parameters are not selected on this ranking.
    performance = pd.read_csv(TABLE_DIR / "Table_1_all_models_evaluation.csv")
    performance = performance.loc[performance["type"] == "ML"].sort_values("RMSE_mL_d")
    order = performance["model"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.0))
    metrics = [("RMSE_mL_d", r"Evaluation RMSE (mL d$^{-1}$)"),
               ("MAE_mL_d", r"Evaluation MAE (mL d$^{-1}$)"),
               ("R2", r"Evaluation $R^2$"), ("TOPSIS", "TOPSIS closeness")]
    for ax, (column, xlabel) in zip(axes.reshape(-1), metrics):
        bars = ax.barh(order, performance[column], color=[MODEL_COLORS.get(n, COLORS["grey"]) for n in order], edgecolor="white", linewidth=.8, alpha=.92)
        for name, bar in zip(order, bars):
            if name == "RandomForest":bar.set_edgecolor(COLORS["ink"]);bar.set_linewidth(2)
        ax.invert_yaxis();style_axis(ax,xlabel,"");numeric_ticks(ax,x=True,y=False,n=5)
        if column in {"R2","TOPSIS"}:ax.set_xlim(0,1.02)
    label_panels(axes)
    save_composite_figure(fig, "Fig03_validation_selection", axes)

'''+source[end:]
exec(compile(source,__file__,'exec'),a.__dict__)
a.create_figures(*joblib.load(a.AUDIT_DIR/'figure_inputs.joblib'))
(QA/'figure_layout_checks.json').write_text(json.dumps(records,indent=2),encoding='utf8')
