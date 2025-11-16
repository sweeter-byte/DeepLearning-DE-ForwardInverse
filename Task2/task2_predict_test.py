#!/usr/bin/env python3
"""
Task 2 Prediction Script (Directly Write to Excel)

- Copy test.xlsx -> test_task2.xlsx (backup if exists)
- Create/replace sheets "k=100", "k=500", "k=1000"
- Predict u(x,y) for each k and write directly to third column
- Optional: also save CSV previews for inspection
"""

import os
import time
import shutil
from pathlib import Path

import torch
import pandas as pd
import numpy as np

# Import your Task2 model definition
from task2_train import Net

SOURCE_TEST_XLSX = 'test.xlsx'
TARGET_TEST_XLSX = 'test_task2.xlsx'
K_VALUES = [100, 500, 1000]
MODEL_TEMPLATE = 'results/models/best_k_{k}.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def safe_copy_source(src: str, dst: str):
    """Copy src -> dst. If dst exists, make a timestamped backup first."""
    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.exists():
        raise FileNotFoundError(f"Source Excel not found: {src}")
    if dst_p.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = str(dst_p.with_name(f"{dst_p.stem}_backup_{stamp}{dst_p.suffix}"))
        shutil.move(dst, backup)
        print(f"Existing {dst} moved to backup: {backup}")
    shutil.copy(src, dst)
    print(f"Copied {src} -> {dst}")

def choose_task2_sheet_name(excel_path: str):
    """Heuristic to select source sheet for Task2 test data."""
    xls = pd.ExcelFile(excel_path)
    names = xls.sheet_names
    candidates = ['子任务2', '子任务二', 'Task2', 'Task 2', 'task2', 'task_2']
    for cand in candidates:
        if cand in names:
            return cand
    return names[0]  # fallback

def create_sheets(dst_xlsx: str, df_source: pd.DataFrame):
    """Create/replace three sheets for k values."""
    with pd.ExcelWriter(dst_xlsx, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for k in K_VALUES:
            df_source.to_excel(writer, sheet_name=f'k={k}', index=False)
            print(f"Created/updated sheet 'k={k}' with {len(df_source)} rows.")

def load_model(k: int):
    path = MODEL_TEMPLATE.format(k=k)
    if not Path(path).exists():
        print(f"Warning: model file not found for k={k}")
        return None
    model = Net(layers=[2,100,100,100,100,1], omega_0=30.0).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def predict_sheet(k: int, dst_xlsx: str):
    sheet_name = f'k={k}'
    df = pd.read_excel(dst_xlsx, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError(f"Sheet {sheet_name} must have at least 2 columns for x,y.")
    model = load_model(k)
    if model is None:
        return
    xy = torch.tensor(df.iloc[:, :2].values.astype(np.float32), device=DEVICE)
    with torch.no_grad():
        u_pred = model(xy).cpu().numpy().squeeze()
    # Write to third column (new column if needed)
    if df.shape[1] < 3:
        df['u_pred'] = u_pred
    else:
        df.iloc[:,2] = u_pred
    # Save sheet back to Excel
    with pd.ExcelWriter(dst_xlsx, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Predictions written to sheet '{sheet_name}' in {dst_xlsx}.")

def main():
    safe_copy_source(SOURCE_TEST_XLSX, TARGET_TEST_XLSX)
    src_sheet = choose_task2_sheet_name(SOURCE_TEST_XLSX)
    df_source = pd.read_excel(SOURCE_TEST_XLSX, sheet_name=src_sheet)
    create_sheets(TARGET_TEST_XLSX, df_source)
    for k in K_VALUES:
        predict_sheet(k, TARGET_TEST_XLSX)
    print("All predictions done. Updated Excel:", TARGET_TEST_XLSX)

if __name__ == "__main__":
    main()
