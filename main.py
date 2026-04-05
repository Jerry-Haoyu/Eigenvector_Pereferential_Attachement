from epam.simulator import EPAMEnsemble

def main():
   E = EPAMEnsemble(N=10, T=5000, m=20, output_path="data/tmp/ensemble_test")
   E.ensemble_simulate()

if __name__ == "__main__":
    main()
