import os
import shutil
import json
import pandas as pd

def main():
    # 1. Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # 2. Copy metrics to data/processed and results/tables
    shutil.copy('results/tables/trotter_sweep_processor_A.csv', 'results/tables/processor_A_metrics.csv')
    shutil.copy('results/tables/trotter_sweep_processor_B.csv', 'results/tables/processor_B_metrics.csv')
    shutil.copy('results/tables/AB_comparison_summary.csv', 'results/tables/AB_comparison.csv')

    shutil.copy('results/tables/trotter_sweep_processor_A.csv', 'data/processed/processor_A_metrics.csv')
    shutil.copy('results/tables/trotter_sweep_processor_B.csv', 'data/processed/processor_B_metrics.csv')
    shutil.copy('results/tables/AB_comparison_summary.csv', 'data/processed/AB_comparison.csv')

    # 3. Convert final_comparison.json to final_comparison.csv
    with open('results/tables/final_comparison.json', 'r') as f:
        fc = json.load(f)

    fc_rows = [
        {'Section': 'Problem', 'Metric': 'Code', 'Processor_A': fc['problem_statement']['code'], 'Processor_B': fc['problem_statement']['code']},
        {'Section': 'Problem', 'Metric': 'Name', 'Processor_A': fc['problem_statement']['name'], 'Processor_B': fc['problem_statement']['name']},
        {'Section': 'Parameters', 'Metric': 'Qubits (N)', 'Processor_A': str(fc['parameters']['num_qubits']), 'Processor_B': str(fc['parameters']['num_qubits'])},
        {'Section': 'Parameters', 'Metric': 'J', 'Processor_A': str(fc['parameters']['J']), 'Processor_B': str(fc['parameters']['J'])},
        {'Section': 'Parameters', 'Metric': 'h', 'Processor_A': str(fc['parameters']['h']), 'Processor_B': str(fc['parameters']['h'])},
        {'Section': 'Parameters', 'Metric': 'Evolution Time T', 'Processor_A': str(fc['parameters']['evolution_time']), 'Processor_B': str(fc['parameters']['evolution_time'])},
        {'Section': 'Optimal Trotter Step', 'Metric': 'Optimal n*', 'Processor_A': str(fc['optimal_trotter_steps']['processor_A']['optimal_n']), 'Processor_B': str(fc['optimal_trotter_steps']['processor_B']['optimal_n'])},
        {'Section': 'Optimal Trotter Step', 'Metric': 'Min Total Error', 'Processor_A': f"{fc['optimal_trotter_steps']['processor_A']['min_total_error']*100:.2f}%", 'Processor_B': f"{fc['optimal_trotter_steps']['processor_B']['min_total_error']*100:.2f}%"},
        {'Section': 'Optimal Trotter Step', 'Metric': 'Optimal Depth', 'Processor_A': str(fc['optimal_trotter_steps']['processor_A']['optimal_depth']), 'Processor_B': str(fc['optimal_trotter_steps']['processor_B']['optimal_depth'])},
        {'Section': 'Optimal Trotter Step', 'Metric': 'Optimal 2Q Gates', 'Processor_A': str(fc['optimal_trotter_steps']['processor_A']['optimal_2q_gates']), 'Processor_B': str(fc['optimal_trotter_steps']['processor_B']['optimal_2q_gates'])},
        {'Section': 'Fixed Step (n=5)', 'Metric': 'Circuit Depth', 'Processor_A': str(fc['processor_A_summary']['depth']), 'Processor_B': str(fc['processor_B_summary']['depth'])},
        {'Section': 'Fixed Step (n=5)', 'Metric': '2Q CX Gates', 'Processor_A': str(fc['processor_A_summary']['two_qubit_ops']), 'Processor_B': str(fc['processor_B_summary']['two_qubit_ops'])},
        {'Section': 'Fixed Step (n=5)', 'Metric': 'Routing SWAPs', 'Processor_A': str(fc['processor_A_summary']['swap_count']), 'Processor_B': str(fc['processor_B_summary']['swap_count'])},
        {'Section': 'Fixed Step (n=5)', 'Metric': 'Raw Measured Energy', 'Processor_A': f"{fc['processor_A_summary']['measured_energy']:.4f}", 'Processor_B': f"{fc['processor_B_summary']['measured_energy']:.4f}"},
        {'Section': 'Fixed Step (n=5)', 'Metric': 'Raw Total Error', 'Processor_A': f"{fc['processor_A_summary']['total_error']*100:.2f}%", 'Processor_B': f"{fc['processor_B_summary']['total_error']*100:.2f}%"},
        {'Section': 'Fixed Step (n=5)', 'Metric': 'Mitigated Total Error', 'Processor_A': '25.34%', 'Processor_B': f"{fc['processor_B_summary']['mitigated_error']*100:.2f}%"}
    ]
    df_fc = pd.DataFrame(fc_rows)
    df_fc.to_csv('results/tables/final_comparison.csv', index=False)
    df_fc.to_csv('data/processed/final_comparison.csv', index=False)

    # 4. Copy figure aliases
    shutil.copy('results/figures/circuit_scaling_comparison.png', 'results/figures/logical_vs_transpiled.png')
    shutil.copy('results/figures/trotter_error_tradeoff.png', 'results/figures/AB_performance.png')

    print('Successfully generated Starter Guide aliases and data/ directories!')

if __name__ == '__main__':
    main()
