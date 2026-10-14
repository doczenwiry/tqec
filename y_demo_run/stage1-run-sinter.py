from tqec.orchestration import simulate_batch

if __name__ == "__main__":
    r = simulate_batch("batch")
    print(r.aggregate, len(r.results), "results,", len(r.failures), "failures")