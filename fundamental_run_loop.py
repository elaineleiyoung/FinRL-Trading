import argparse
import subprocess
if __name__ == '__main__':
    sector_list = [0] + list(range(10, 65, 5))
    # sector_list.remove(10)
    python_env = "venv\\Scripts\\python.exe"
    for sector in sector_list:
        print(f"Running for sector{sector}...")
        subprocess.run([
            python_env, 'fundamental_run_model.py',
            '--sector_name_input', f'sector{sector}',
            '--fundamental_input', 'data_processor_update/outputs/final_ratios.csv',
            '--sector_input', f'data_processor_update/outputs/sector{sector}.xlsx'
        ])
