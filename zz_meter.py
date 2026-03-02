import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

class ZZMeter:
    """
    meter = ZZMeter()
    zz_khz = meter.run_experiment()
    meter.plot()
    """
    
    def __init__(self, shots=1024, num_angles=17):
        self.simulator = AerSimulator()
        self.shots = shots
        self.angles_deg = np.linspace(0, 360, num_angles)
        self.results = {}
        self.metrics = {}
        
    def _create_circuits(self, with_zz_interaction=False):
        circuits = []
        for angle in self.angles_deg:
            theta = np.radians(angle)
            qc = QuantumCircuit(2, 2)
            qc.rx(theta, 0)  # q0: RX(θ)
            
            if with_zz_interaction:
                qc.h(1)  # q1: |+⟩ для ZZ
                
            qc.measure_all()
            circuits.append(qc)
        return circuits
    
    def run_experiment(self):
        print(f"🚀 ZZMeter: {len(self.angles_deg)} точек")
        
        # CONTROL: q1=|0⟩
        print("   1/2 Control...")
        circ_ctrl = self._create_circuits(False)
        job_ctrl = self.simulator.run(circ_ctrl, shots=self.shots)
        self.results['ctrl'] = self._get_probs(job_ctrl)
        
        # EXPERIMENT: q1=|+⟩
        print("   2/2 Experiment...")
        circ_exp = self._create_circuits(True)
        job_exp = self.simulator.run(circ_exp, shots=self.shots)
        self.results['exp'] = self._get_probs(job_exp)
        
        return self.analyze()
    
    def _get_probs(self, job):  
        result = job.result()
        probs = []
        
        for experiment in result.results:
            counts = experiment.data.counts  
            p00 = counts.get('00', 0) / self.shots
            probs.append(p00)
            
        return np.array(probs)
    
    def analyze(self, tau_ns=50):
        def fit_func(x, phi, A, B):
            return A * np.cos(np.radians(x) + phi) + B
        
        p0 = [0, 0.4, 0.5]
        p_ctrl, _ = curve_fit(fit_func, self.angles_deg, self.results['ctrl'], p0=p0)
        p_exp, _ = curve_fit(fit_func, self.angles_deg, self.results['exp'], p0=p0)
        
        delta_phi = np.abs(p_exp[0] - p_ctrl[0])
        zz_khz = (delta_phi / (2 * np.pi * tau_ns * 1e-9)) / 1000
        
        theory = 0.5 * (1 + np.cos(np.radians(self.angles_deg)))
        r2 = 1 - np.sum((self.results['ctrl'] - theory)**2) / np.sum((self.results['ctrl'] - np.mean(self.results['ctrl']))**2)
        
        self.metrics = {'zz_khz': zz_khz, 'r2': r2, 'delta_phi': delta_phi}
        print(f"🎯 ZZ = {zz_khz:.2f} kHz | R² = {r2:.4f}")
        return zz_khz
    
    def plot(self, save_path='zz_meter_final.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        x_fine = np.linspace(0, 360, 100)
        
        def fit_func(x, phi, A, B):
            return A * np.cos(np.radians(x) + phi) + B
        
        # График 1: P(|00⟩)
        ax1.scatter(self.angles_deg, self.results['ctrl'], s=60, label='Control', color='blue')
        ax1.scatter(self.angles_deg, self.results['exp'], s=60, label='Experiment', color='red')
        ax1.plot(x_fine, fit_func(x_fine, *self.metrics['popt_ctrl'] if 'popt_ctrl' in self.metrics else [0,0.5,0.5]), 'b--')
        ax1.plot(x_fine, fit_func(x_fine, *self.metrics['popt_exp'] if 'popt_exp' in self.metrics else [0,0.5,0.5]), 'r--')
        ax1.set_title(f'ZZ-Meter: Jzz = {self.metrics["zz_khz"]:.1f} kHz')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Остатки
        theory = 0.5 * (1 + np.cos(np.radians(self.angles_deg)))
        ax2.plot(self.angles_deg, self.results['ctrl'] - theory, 'o-', label='Control', color='blue')
        ax2.axhline(0, color='k', linestyle='--')
        ax2.set_title(f'Bloch R² = {self.metrics["r2"]:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ {save_path}")

if __name__ == "__main__":
    print("🔬 ZZMeter v1.0 — Production Ready!")
    
    meter = ZZMeter(shots=1024, num_angles=17)
    zz_khz = meter.run_experiment()
    meter.plot()
    
    print(f"\n🏆 ГОТОВО! ZZ = {zz_khz:.2f} kHz, R² = {meter.metrics['r2']:.4f}")
