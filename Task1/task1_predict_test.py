"""
Task 1 Prediction Script
Predict u(x, y) values and directly update '子任务1' sheet in test.xlsx
"""

import torch
import pandas as pd
from pathlib import Path
import shutil
from task1_train import Net  # 使用你提供的 Task 1 模型类

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def load_model(checkpoint_path, layers=[2, 100, 100, 100, 100, 1], omega_0=30.0):
    """Load trained Task 1 model"""
    print(f"Loading model: {checkpoint_path}")
    
    model = Net(layers=layers, omega_0=omega_0).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model


def predict_u(model, df_test):
    """Predict u(x, y) values from dataframe"""
    x = torch.tensor(df_test.iloc[:, 0].values, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(df_test.iloc[:, 1].values, dtype=torch.float32, device=DEVICE)
    xy = torch.stack([x, y], dim=1)

    with torch.no_grad():
        u_pred = model(xy).cpu().numpy()

    print(f"✓ Prediction done: u(x,y) min={u_pred.min():.6f}, max={u_pred.max():.6f}, mean={u_pred.mean():.6f}")
    return u_pred


def fill_task1_sheet(model, excel_path):
    """
    Fill the '子任务1' sheet in-place with predicted u(x,y) values
    """
    print(f"\nUpdating Excel file: {excel_path}")
    backup_file = excel_path.replace('.xlsx', '_backup.xlsx')
    shutil.copy(excel_path, backup_file)
    print(f"✓ Backup created: {backup_file}")

    # Load Excel sheet
    df_task1 = pd.read_excel(excel_path, sheet_name='子任务1')
    print(f"Loaded '子任务1' sheet: {len(df_task1)} rows, columns: {df_task1.columns.tolist()}")

    # Predict
    u_pred = predict_u(model, df_task1)

    # Fill predicted values in the third column (index 2)
    df_task1.iloc[:, 2] = u_pred

    # Save CSV preview
    preview_csv = excel_path.replace('.xlsx', '_task1_preview.csv')
    df_task1.to_csv(preview_csv, index=False)
    print(f"✓ Preview CSV saved: {preview_csv}")

    # Save back to Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_task1.to_excel(writer, sheet_name='子任务1', index=False)
    print(f"✓ Excel updated successfully: {excel_path}")


def main():
    model_path = 'results/models/best_model.pth'
    test_file = 'test.xlsx'

    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return

    # Load model
    model = load_model(model_path)

    # Fill Excel directly
    fill_task1_sheet(model, test_file)

    print("\n✅ Task 1 prediction and Excel update completed!")


if __name__ == '__main__':
    main()
