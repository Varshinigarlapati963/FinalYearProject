"""Save Q-BPSO v2 results directly from logged output data."""
import pandas as pd, os

rows = [
    # Dataset-1
    {"Dataset":"dataset1","Method":"Q-BPSO v2","N_Original":1490,"N_Selected":334,"Reduction_pct":77.6,"Best_Fitness":0.841313,"QBPSO_Time_s":446.2,"Model":"KNN",         "Accuracy":88.7683,"Precision":88.7916,"Recall":88.7683,"F1-Score":88.7669,"Train_Time_s":None,"Inference_Time_s":None},
    {"Dataset":"dataset1","Method":"Q-BPSO v2","N_Original":1490,"N_Selected":334,"Reduction_pct":77.6,"Best_Fitness":0.841313,"QBPSO_Time_s":446.2,"Model":"GaussianNB",   "Accuracy":53.6952,"Precision":71.4776,"Recall":53.6952,"F1-Score":41.7105,"Train_Time_s":None,"Inference_Time_s":None},
    {"Dataset":"dataset1","Method":"Q-BPSO v2","N_Original":1490,"N_Selected":334,"Reduction_pct":77.6,"Best_Fitness":0.841313,"QBPSO_Time_s":446.2,"Model":"DecisionTree", "Accuracy":89.3111,"Precision":89.3473,"Recall":89.3111,"F1-Score":89.3090,"Train_Time_s":None,"Inference_Time_s":None},
    {"Dataset":"dataset1","Method":"Q-BPSO v2","N_Original":1490,"N_Selected":334,"Reduction_pct":77.6,"Best_Fitness":0.841313,"QBPSO_Time_s":446.2,"Model":"RandomForest", "Accuracy":89.9374,"Precision":89.9552,"Recall":89.9374,"F1-Score":89.9365,"Train_Time_s":None,"Inference_Time_s":None},
    # Dataset-2
    {"Dataset":"dataset2","Method":"Q-BPSO v2","N_Original":173,"N_Selected":21,"Reduction_pct":87.9,"Best_Fitness":0.635022,"QBPSO_Time_s":205.3,"Model":"KNN",         "Accuracy":43.4333,"Precision":53.8716,"Recall":43.4333,"F1-Score":43.6339,"Train_Time_s":None,"Inference_Time_s":None},
    {"Dataset":"dataset2","Method":"Q-BPSO v2","N_Original":173,"N_Selected":21,"Reduction_pct":87.9,"Best_Fitness":0.635022,"QBPSO_Time_s":205.3,"Model":"GaussianNB",   "Accuracy":34.0500,"Precision":58.3848,"Recall":34.0500,"F1-Score":19.0030,"Train_Time_s":None,"Inference_Time_s":None},
    {"Dataset":"dataset2","Method":"Q-BPSO v2","N_Original":173,"N_Selected":21,"Reduction_pct":87.9,"Best_Fitness":0.635022,"QBPSO_Time_s":205.3,"Model":"DecisionTree", "Accuracy":67.2833,"Precision":64.7001,"Recall":67.2833,"F1-Score":56.9886,"Train_Time_s":None,"Inference_Time_s":None},
    {"Dataset":"dataset2","Method":"Q-BPSO v2","N_Original":173,"N_Selected":21,"Reduction_pct":87.9,"Best_Fitness":0.635022,"QBPSO_Time_s":205.3,"Model":"RandomForest", "Accuracy":67.2833,"Precision":64.8592,"Recall":67.2833,"F1-Score":56.8429,"Train_Time_s":None,"Inference_Time_s":None},
    # Dataset-3
    {"Dataset":"dataset3","Method":"Q-BPSO v2","N_Original":330,"N_Selected":75,"Reduction_pct":77.3,"Best_Fitness":0.893825,"QBPSO_Time_s":65.3,"Model":"KNN",         "Accuracy":90.00,"Precision":90.4040,"Recall":90.00,"F1-Score":89.9749,"Train_Time_s":None,"Inference_Time_s":None},
    {"Dataset":"dataset3","Method":"Q-BPSO v2","N_Original":330,"N_Selected":75,"Reduction_pct":77.3,"Best_Fitness":0.893825,"QBPSO_Time_s":65.3,"Model":"GaussianNB",   "Accuracy":81.25,"Precision":84.9406,"Recall":81.25,"F1-Score":80.7415,"Train_Time_s":None,"Inference_Time_s":None},
    {"Dataset":"dataset3","Method":"Q-BPSO v2","N_Original":330,"N_Selected":75,"Reduction_pct":77.3,"Best_Fitness":0.893825,"QBPSO_Time_s":65.3,"Model":"DecisionTree", "Accuracy":93.75,"Precision":93.7774,"Recall":93.75,"F1-Score":93.7490,"Train_Time_s":None,"Inference_Time_s":None},
    {"Dataset":"dataset3","Method":"Q-BPSO v2","N_Original":330,"N_Selected":75,"Reduction_pct":77.3,"Best_Fitness":0.893825,"QBPSO_Time_s":65.3,"Model":"RandomForest", "Accuracy":95.00,"Precision":95.1128,"Recall":95.00,"F1-Score":94.9969,"Train_Time_s":None,"Inference_Time_s":None},
]

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(rows)
out = os.path.join("results", "quantum_swarm_results_v2.csv")
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
print(df[df["Model"]=="RandomForest"][["Dataset","N_Selected","Accuracy","F1-Score","QBPSO_Time_s"]].to_string(index=False))
