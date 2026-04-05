import unittest 
import numpy as np
from epam.simulator import Simulator
from scipy.sparse.linalg import svds

class testSimulator(unittest.TestCase):
    def test_basic(self):
        for m in range(1, 5):
            print(f"--------------testing m = {m} case------------------")
            model = Simulator(T=5, N=1, output_path="data/tmp", m=m)
            #----------- check initalization -------
            A_clique = np.ones([m+1, m+1]) - np.eye(m+1)
            
            self.assertTrue(np.linalg.norm(A_clique - model.A.get_csr()[:m+1, :m+1],ord=2) < 1e-10)

            
            model.simulate()
            #check shape
            N = model.A.shape 
            self.assertEqual(N, 5+m+1)
            
            #-------- check simulation result --------
            #check if A is filled 
            self.assertTrue(model.A.get_csr()[5+m, 5+m] != None)
            
            #check no self-loop
            self.assertAlmostEqual(np.trace(model.A.get_csr().toarray()), 0.0, delta=1e-10)
    def test_no_negative(self):
        
            
            
if __name__ == '__main__':
    print("-----testing simulator----")
    unittest.main()

            