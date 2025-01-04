import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# TCN dilation patterns
patterns = {
    'TCN_Default': lambda n: [2**i for i in range(n)],  # Classic TCN [1,2,4,8,...]
    'Exponential': lambda n: [2**i for i in range(n)],  # [1,2,4,8,16,...]
    'Fibonacci': lambda n: [1, 2] + [sum(([1, 2] + [sum(([1, 2] + [sum(([1, 2][i-2:i])) for i in range(2, j)]))
                           for i in range(2, j-2)])) for j in range(4, n+2)],  # [1,2,3,5,8,13,...]
    'Hybrid': lambda n: [2**i for i in range(n//2)] + [i+1 for i in range(n//2, n)],  # [1,2,4,8,5,6,7,8]
    'Linear': lambda n: [i+1 for i in range(n)],  # [1,2,3,4,...]
    'Pyramid': lambda n: ([2**i for i in range(n//2)] + 
                         [2**(n//2-i-1) for i in range(n-n//2)]),  # [1,2,4,2,1] for n=5
    'Smoothed': lambda n: [min(2**i, max(1, int(2**(n//2) * (1 - 0.3 * max(0, i - n//2 + 1))))) 
                          for i in range(n)],  # [1,2,4,6,7,7,6,5] smoothed shoulder
    'ASPP': lambda n: [1] + [2**(i-1) for i in range(1, min(4, n))] + 
                     [2**3]*(max(0, n-4))  # [1,1,2,4,8,8,8,...] ASPP-style
}

def calculate_rf_size(kernel_size, dilation_levels):
    """Calculate receptive field size for TCN architecture"""
    rf = 1
    for d in dilation_levels:
        rf += (kernel_size - 1) * d
    return rf

def compare_dilation_patterns(sequence_length=1000, kernel_sizes=[2,3], num_levels=4):
    """Compare different TCN dilation patterns and their receptive fields"""
    results = {}
    for kernel_size, (pattern_name, pattern_fn) in product(kernel_sizes, patterns.items()):
        dilation_levels = pattern_fn(num_levels)
        rf_size = min(calculate_rf_size(kernel_size, dilation_levels), sequence_length)
        coverage = rf_size / sequence_length * 100
        results[(kernel_size, pattern_name)] = {
            'rf_size': rf_size,
            'coverage': coverage,
            'dilation_levels': dilation_levels
        }
    return results

def plot_comparison(results, sequence_length=1000):
    """Plot comparison of different dilation patterns"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    patterns = list(set(p for _, p in results.keys()))
    kernel_sizes = list(set(k for k, _ in results.keys()))
    x = np.arange(len(patterns))
    width = 0.8 / len(kernel_sizes)
    
    for i, k in enumerate(kernel_sizes):
        rf_sizes = [results[(k, p)]['rf_size'] for p in patterns]
        ax1.bar(x + i*width, rf_sizes, width, label=f'Kernel={k}')
        coverage = [results[(k, p)]['coverage'] for p in patterns]
        ax2.bar(x + i*width, coverage, width, label=f'Kernel={k}')
    
    for ax in [ax1, ax2]:
        ax.set_xticks(x + width * (len(kernel_sizes)-1)/2)
        ax.set_xticklabels(patterns, rotation=45)
        ax.legend()
    
    ax1.set_ylabel('Receptive Field Size')
    ax1.set_title('Receptive Field Size Comparison')
    ax2.set_ylabel('Sequence Coverage (%)')
    ax2.set_title('Sequence Coverage Comparison')
    
    plt.tight_layout()
    return fig

def visualize_dilation_pattern(pattern_name, dilation_levels):
    """Visualize how the dilation pattern grows"""
    plt.figure(figsize=(10, 3))
    plt.plot(dilation_levels, 'o-', label=f'Dilation rates')
    plt.title(f'{pattern_name} Dilation Pattern')
    plt.xlabel('Layer')
    plt.ylabel('Dilation Rate')
    plt.grid(True)
    plt.legend()
    return plt.gcf()

def print_kernel_analysis(pattern_name, dilation_levels):
    """Print receptive field analysis for different kernel sizes"""
    print(f"\n{pattern_name} dilations:", dilation_levels)
    print("Kernel size | Receptive Field")
    print("-" * 30)
    for k in range(2, 10):  # kernel sizes 2-9
        rf = calculate_rf_size(k, dilation_levels)
        print(f"    {k}     |     {rf}")
    print("-" * 30)

def main():
    # Test sequence lengths 3, 4, and 8
    sequence_lengths = [3, 4, 8]
    kernel_sizes = list(range(2, 10))  # [2,3,4,5,6,7,8,9]
    num_levels = 8
    
    for seq_len in sequence_lengths:
        print(f"\nAnalyzing sequence length: {seq_len}")
        print("=" * 50)
        
        # Compare patterns
        results = compare_dilation_patterns(seq_len, kernel_sizes, num_levels)
        
        # Plot comparisons
        fig = plot_comparison(results)
        fig.savefig(f'dilation_pattern_comparison_len{seq_len}.png')
        plt.close(fig)
        
        # Visualize individual patterns and analyze different kernel sizes
        for pattern_name, pattern_fn in patterns.items():
            dilation_levels = pattern_fn(num_levels)
            fig = visualize_dilation_pattern(pattern_name, dilation_levels)
            fig.savefig(f'dilation_{pattern_name.lower()}_len{seq_len}.png')
            plt.close(fig)
            
            print_kernel_analysis(pattern_name, dilation_levels)

if __name__ == "__main__":
    main()
