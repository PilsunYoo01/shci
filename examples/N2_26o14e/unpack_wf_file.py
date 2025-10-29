import subprocess
import numpy as np

def load_wavefunction_from_tool(wf_filename):
    """Load wavefunction using the existing unpack_wf tool"""
    try:
        # Run the unpack_wf tool
        result = subprocess.run(['/home/_qunova/apps/shci_singleshot/tools/unpack_wf/unpack_wf', wf_filename], 
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        
        determinants = []
        coefficients = []
        
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 3:
                # Parse orbital occupations
                up_orbs = [int(x)-1 for x in parts[0].split()]  # Convert to 0-based indexing
                dn_orbs = [int(x)-1 for x in parts[1].split()]
                
                # Parse coefficients
                coefs = [float(x) for x in parts[2:]]
                
                determinants.append({'up': up_orbs, 'dn': dn_orbs})
                coefficients.append(coefs)
        
        return determinants, np.array(coefficients)
    
    except subprocess.CalledProcessError as e:
        print(f"Error running unpack_wf: {e}")
        return None, None

# Usage
dets, coefs = load_wavefunction_from_tool("wf_eps1_1.00e-05.dat")
if dets is not None:
    print(f"Loaded {len(dets)} determinants")
    print(f"{dets[:10]}")
    print(f"Ground state CI vector shape: {coefs.shape}")
    print(f"First few coefficients: {coefs[:5, 0]}")
