import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def clean_dataset2():
    # Construct path to the Excel file
    excel_path = os.path.join("data", "data to be upload.xlsx")
    if not os.path.exists(excel_path):
        excel_path = "data to be upload.xlsx"
        
    print(f"Loading Excel file from: {excel_path}")
    
    # 1. Read Excel file using pd.ExcelFile() and print sheet names
    excel = pd.ExcelFile(excel_path)
    sheet_names = excel.sheet_names
    print(f"Sheet names in Excel file: {sheet_names}")
    
    target_sheet = None
    
    # 2. Iterate through each sheet and check if Column A contains 'MALWARE' marker
    for sheet in sheet_names:
        temp_df = pd.read_excel(excel, sheet_name=sheet, header=None)
        col_a = temp_df.iloc[:, 0].astype(str).str.strip().str.upper()
        if (col_a == 'MALWARE').any():
            target_sheet = sheet
            print(f"Found sheet containing 'MALWARE' marker: {target_sheet}")
            break

    metadata_cols = [
        'App', 'Package', 'Category', 'Description', 'Rating', 
        'Number of ratings', 'Price', 'Related apps', 
        'Dangerous permissions count', 'Safe permissions count'
    ]

    # 3. Process according to marker structure or separate BENIGN/MALWARE sheets
    if target_sheet is not None:
        df_raw = pd.read_excel(excel, sheet_name=target_sheet)
        col_a = df_raw.iloc[:, 0].astype(str).str.strip().str.upper()
        
        # Find index of MALWARE marker
        malware_indices = df_raw[col_a == 'MALWARE'].index
        malware_idx = malware_indices[0] if len(malware_indices) > 0 else len(df_raw)
        
        # 4. Create Label column: 0 for rows above MALWARE marker, 1 for rows below
        labels = [0 if i < malware_idx else 1 for i in range(len(df_raw))]
        df_raw = df_raw.assign(Label=labels)
        
        # 5. Drop rows containing 'BENIGN' and 'MALWARE' in Column A
        df_clean = df_raw[~col_a.isin(['BENIGN', 'MALWARE'])].copy()
    else:
        print("Checking sheet structure for BENIGN and MALWARE datasets...")
        # Fallback for workbook split into BENIGN and MALWARE sheets
        if 'BENIGN' in sheet_names and 'MALWARE' in sheet_names:
            df_benign = pd.read_excel(excel, 'BENIGN')
            df_malware = pd.read_excel(excel, 'MALWARE', header=None)
            df_malware.columns = df_benign.columns
            
            df_benign = df_benign.assign(Label=0)
            df_malware = df_malware.assign(Label=1)
            
            df_clean = pd.concat([df_benign, df_malware], ignore_index=True)
        else:
            raise ValueError("Could not find BENIGN or MALWARE data in the Excel file.")

    # 6. Drop metadata columns
    cols_to_drop = [col for col in metadata_cols if col in df_clean.columns]
    df_features = df_clean.drop(columns=cols_to_drop)
    
    # 7. Keep permission feature columns (~173 features) plus 'Label' column
    print(f"Final columns count: {df_features.shape[1]}")
    
    # 8. Save cleaned dataframe as dataset2.csv in /data folder
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset2.csv")
    
    df_features.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_path}")
    
    # Print shape and value_counts of Label column
    print("\n--- Output Verification ---")
    print(f"Dataframe Shape: {df_features.shape}")
    print("\nLabel Value Counts:")
    print(df_features['Label'].value_counts())
    
    return df_features

if __name__ == "__main__":
    clean_dataset2()
