# main.py
from pipeline.pipeline import run_pipeline
import time
import tracemalloc
def main():
    print("--------------------\nRunning pipeline...\n--------------------")
    run_pipeline()
    print("--------------------\nPipeline execution completed.\n--------------------")

if __name__ == "__main__":
    tracemalloc.start()
    start_time = time.perf_counter()
    main()
    # 3. Stop timer and gather memory metrics
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 4. Print results  
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print(f"Current Memory: {current / 10**6:.2f} MB")
    print(f"Peak Memory:    {peak / 10**6:.2f} MB")