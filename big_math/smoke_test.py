import sys
if not hasattr(sys.modules['__main__'], '__spec__'):
    sys.modules['__main__'].__spec__ = None

if __name__ == '__main__':
    import time, numpy as np
    from trace.load_data import load_data
    from trace.rh_model_setting import inference_on_ds, RH_labeling

    model_name = "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_10"

    ds = load_data(cheat=True)[:50]
    gen = inference_on_ds(ds, model_name, save_path="/tmp/test_inference.json")

    true_set, false_set = RH_labeling(gen, model_name, cheat=True)
    print(f"true: {len(true_set)}, false: {len(false_set)}")

    # gradient_h.py expects 'input'/'output' keys; BigMath uses 'prompt'/'gen'
    for ex in true_set + false_set:
        ex['input'] = ex.pop('prompt')
        ex['output'] = ex.pop('gen')

    if len(false_set) == 0:
        print("WARNING: false_set is empty — try step 20 instead")
        print("  model_name = ...global_step_20")

    # vLLM phase done — now safe to initialize CUDA for PyTorch gradient extraction
    import torch
    from icl.gradient.gradient_h import get_gradients_over_dataset, layer_selection
    from icl.gradient.gradient import load_model_and_tokenizer

    def gpu_mem_gb():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def peak_mem_gb():
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model_and_tokenizer(model_name, LORA=True)
    print(f"[mem] after loading model for gradients: {gpu_mem_gb():.1f} GB allocated")

    selected = layer_selection(true_set + false_set, model_name)
    print(f"Selected layers: {selected}")

    start = time.time()
    true_grads, _ = get_gradients_over_dataset(model, tokenizer, true_set[:10], layers=selected)
    false_grads, _ = get_gradients_over_dataset(model, tokenizer, false_set[:min(10, len(false_set))], layers=selected)
    elapsed = time.time() - start
    print(f"[mem] after gradient extraction: peak {peak_mem_gb():.1f} GB")

    true_norms = [np.linalg.norm(g.numpy()) for g in true_grads]
    false_norms = [np.linalg.norm(g.numpy()) for g in false_grads]
    print(f"True grad norm mean:  {np.mean(true_norms):.3f}")
    print(f"False grad norm mean: {np.mean(false_norms):.3f}")

    print(f"10 samples: {elapsed:.1f}s → {elapsed/10:.1f}s/sample")
    print(f"Projected full run (200 samples × 4 checkpoints): {200*4*elapsed/10/3600:.1f} GPU-hours")
