# main.py
from pipeline.pipeline import run_pipeline

def main():
    print("--------------------\nRunning pipeline...\n--------------------")
    run_pipeline()
    print("--------------------\nPipeline execution completed.\n--------------------")

if __name__ == "__main__":
    main()