import onnx
import numpy as np
from onnx import numpy_helper
import difflib

def get_layer_data(tensor):
    arr = numpy_helper.to_array(tensor)
    # We create a "signature" tuple for alignment: (Dimensions)
    # We use shape as the primary key for structural alignment.
    return {
        "name": tensor.name,
        "shape": tuple(arr.shape),
        "val": arr,
        "mean": np.mean(arr),
        "std": np.std(arr),
        "signature": str(tuple(arr.shape)) 
    }

def compare_lcs(path_a, path_b):
    print(f"Loading Models...\n  A: {path_a}\n  B: {path_b}\n")
    
    model_a = onnx.load(path_a)
    model_b = onnx.load(path_b)
    
    # 1. Extract lists
    list_a = [get_layer_data(t) for t in model_a.graph.initializer]
    list_b = [get_layer_data(t) for t in model_b.graph.initializer]

    # 2. Create Signatures for Diff Algorithm
    # We align based on SHAPE. If shapes match, we assume it's a potential match.
    sig_a = [x['signature'] for x in list_a]
    sig_b = [x['signature'] for x in list_b]

    # 3. Compute Diff (Longest Common Subsequence)
    matcher = difflib.SequenceMatcher(None, sig_a, sig_b)
    
    print(f"{'LAYER A':<45} | {'LAYER B':<45} | {'STATUS'}")
    print("-" * 130)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        
        # tag is: 'replace', 'delete', 'insert', 'equal'
        
        if tag == 'equal':
            # These ranges match in structure. Now check values.
            for k in range(i2 - i1):
                la = list_a[i1 + k]
                lb = list_b[j1 + k]
                
                # Check for Value Changes
                diff = np.max(np.abs(la['val'] - lb['val']))
                if diff == 0:
                    status = "IDENTICAL"
                elif diff < 1e-5:
                    status = "~ FLOAT NOISE"
                else:
                    status = f"CHANGED (Diff: {diff:.4f})"
                    # Heuristic for initialization check
                    if abs(la['std']) > 0.05 and abs(lb['std']) < 0.001:
                        status += " [B Untrained?]"
                    elif abs(lb['std']) > 0.05 and abs(la['std']) < 0.001:
                        status += " [A Untrained?]"

                name_a = la['name'][-45:]
                name_b = lb['name'][-45:]
                print(f"{name_a:<45} | {name_b:<45} | {status}")

        elif tag == 'delete':
            # Exists in A, missing in B
            for k in range(i2 - i1):
                la = list_a[i1 + k]
                name_a = la['name'][-45:]
                print(f"{name_a:<45} | {'---':<45} | <<< DELETED IN B {la['shape']}")

        elif tag == 'insert':
            # Exists in B, missing in A
            for k in range(j2 - j1):
                lb = list_b[j1 + k]
                name_b = lb['name'][-45:]
                print(f"{'---':<45} | {name_b:<45} | >>> INSERTED IN B {lb['shape']}")

        elif tag == 'replace':
            # Structural Mismatch block
            len_a = i2 - i1
            len_b = j2 - j1
            
            # Print side-by-side as best as we can
            max_len = max(len_a, len_b)
            for k in range(max_len):
                str_a = ""
                str_b = ""
                
                if k < len_a:
                    la = list_a[i1 + k]
                    str_a = f"{la['name'][-30:]} {la['shape']}"
                
                if k < len_b:
                    lb = list_b[j1 + k]
                    str_b = f"{lb['name'][-30:]} {lb['shape']}"
                
                print(f"{str_a:<45} | {str_b:<45} | !!! SHAPE MISMATCH BLOCK")

if __name__ == "__main__":
    # --- UPDATE FILES ---
    file1 = "/cs/projects/MWNaturalPredict/DL/predSWR/frozen_models/ES_model_tripletRemake1334.onnx"
    # file1 = "/cs/projects/MWNaturalPredict/DL/predSWR/frozen_models/model_tripletRemake1334_1Batch.onnx"
    file2 = "/cs/projects/MWNaturalPredict/DL/predSWR/frozen_models/model.onnx"
    # file2 = "/cs/projects/MWNaturalPredict/DL/predSWR/frozen_models/model_2.onnx"
    
    compare_lcs(file1, file2)