"""
Task 3: Predict k(x,y) values in test.xlsx (directly updates the file)
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from task3_train import PINN_Task3
import shutil

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def load_model(checkpoint_path, u_layers=[2, 64, 64, 64, 1], k_layers=[2, 32, 32, 32, 1]):
    """Load trained model"""
    print(f"Loading model: {checkpoint_path}")
    
    model = PINN_Task3(u_layers=u_layers, k_layers=k_layers).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully (Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']:.6e})")
    return model


def predict_k(model, df_test):
    """Predict k(x,y) from dataframe"""
    x_test = torch.tensor(df_test.iloc[:, 0].values, dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(df_test.iloc[:, 1].values, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        k_pred = model.forward_k(x_test, y_test).cpu().numpy()
        u_pred = model.forward_u(x_test, y_test).cpu().numpy()  # optional

    print(f"✓ Prediction done")
    print(f"k(x,y) stats: min={k_pred.min():.6f}, max={k_pred.max():.6f}, mean={k_pred.mean():.6f}")
    return k_pred, u_pred


def fill_task3_sheet(model, excel_path):
    """
    Fill the '子任务3' sheet in-place with predicted k values
    """
    print(f"\nUpdating Excel file: {excel_path}")
    backup_file = excel_path.replace('.xlsx', '_backup.xlsx')
    shutil.copy(excel_path, backup_file)
    print(f"✓ Backup created: {backup_file}")

    # Load Excel and find sheet '子任务3'
    excel_file = pd.ExcelFile(excel_path)
    if "子任务3" not in excel_file.sheet_names:
        raise ValueError("❌ Sheet '子任务3' not found in Excel file!")

    df_task3 = pd.read_excel(excel_path, sheet_name="子任务3")
    print(f"Task 3 sheet loaded: {len(df_task3)} rows, columns: {df_task3.columns.tolist()}")

    # Predict
    k_pred, u_pred = predict_k(model, df_task3)

    # Fill k values in third column (index 2)
    df_task3.iloc[:, 2] = k_pred

    # Save preview CSV for inspection
    preview_csv = excel_path.replace('.xlsx', '_task3_preview.csv')
    df_task3.to_csv(preview_csv, index=False)
    print(f"✓ Preview CSV saved: {preview_csv}")

    # Save back to Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_task3.to_excel(writer, sheet_name="子任务3", index=False)
    print(f"✓ Excel updated successfully: {excel_path}")


def main():
    print("="*70)
    print("Task 3: Predict k(x,y) and fill Excel sheet")
    print("="*70)

    model_path = 'results/best_model.pth'
    test_file = 'test.xlsx'

    # Check files
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return

    # Load model
    model = load_model(model_path)

    # Fill Excel directly
    fill_task3_sheet(model, test_file)

    print("\n" + "="*70)
    print("✅ Prediction and Excel update completed!")
    print("="*70)


if __name__ == '__main__':
    main()
