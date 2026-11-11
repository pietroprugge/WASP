import importlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import setuptools  # noqa: F401  # Provides distutils compatibility for batman.

batman = importlib.import_module("batman")

class Transit:

    """
    Transit model class for exoplanet detection using batman.
    
    This class generates synthetic light curves for exoplanet transits
    and provides visualization capabilities.
    """
    
    def __init__(self, params_dict):

        """
        Initialize the transit model with parameters from configuration file.
        
        Parameters
        ----------
        params_dict : dict
            Dictionary containing transit parameters loaded from YAML file
        """
        self.params = batman.TransitParams()
        self.params.t0 = params_dict.get('t0', 0.0)
        self.params.per = params_dict.get('per')
        self.params.rp = params_dict.get('rp')
        self.params.a = params_dict.get('a')
        self.params.inc = params_dict.get('inc')
        self.params.ecc = params_dict.get('ecc', 0.0)
        self.params.w = params_dict.get('w', 90.0)
        self.params.u = params_dict.get('u', [0.1, 0.3])
        self.params.limb_dark = params_dict.get('limb_dark', 'quadratic')
        
        # Time array for model evaluation
        self.t = np.linspace(-0.08, 0.08, 1000)
        
        # Initialize batman model, considering also an error of 0.5
        self.model = batman.TransitModel(self.params, self.t, max_err = 0.5)

    def compute_light_curve(self):
        """
        Compute the light curve using the batman transit model.
        
        Returns
        -------
        flux : ndarray
            Array of relative flux values
        """
        self.flux = self.model.light_curve(self.params)
        return self.flux

    def plot_light_curve(self,name_files, output_file='wasp39b', show=False, one_output = False):
        """
        Plot and save the transit light curve.
        
        Parameters
        ----------
        output_file : str, optional
            Output filename for the plot (default: 'wasp39b')
        """
        if not hasattr(self, 'flux'):
            self.compute_light_curve()
        plt.figure(figsize=(8, 5))
        plt.plot(self.t * 24, self.flux, label=f'Batman model {name_files[0]}', color='navy', lw=2)
        plt.xlabel("Hours from mid-transit")
        plt.ylabel("Normalized Flux")
        plt.title(f"Theoretical Light Curve {name_files[0]} ")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        if (one_output == False):
            plt.savefig(output_file, dpi=300) 
        if show:
            plt.show()
        print(f"Light curve saved to {output_file}")
    
    def run(self, output_file='wasp39b_assignment1_taskF'):
        """
        Run the complete transit model workflow: compute and plot light curve.
        
        Parameters
        ----------
        output_file : str, optional
            Output filename for the plot (default: 'wasp39b_assignment1_taskF')
        """
        self.compute_light_curve()
        self.plot_light_curve(output_file)

    def error (self):
        """
        computing  the truncation error in case we are using  a “square-root”, “logarithmic”, “exponential”, “nonlinear” or “custom” 
        limb darkening model 

        this happens because it's only valid for methods that use a numerical integration
        then once it's computed through the batman model we can plot the final result

        """
        plt.figure()
        if (self.params.limb_dark == 'quadratic' or self.params.limb_dark == 'uniform' or self.params.limb_dark == 'linear'):
            print("no error for a",self.params.limb_dark,"limb darkening model")
        else:
            err = self.model.calc_err(plot = True)
            return err

    #If we want to create a single plot
    def plot_one_figure(self, flux,name_files, output_file='wasp39b', show=False):
            """
            Plot and save the transit light curve for multiple planets
            
            Parameters
            ----------
            output_file : str, optional
                Output filename for the plot (default: 'wasp39b')
            """
            plt.figure(figsize=(8, 5))
            for i in range(len(flux)):
                plt.plot(self.t * 24, flux[i], label=f'Batman model {name_files[i]}', lw=2)
            plt.xlabel("Hours from mid-transit")
            plt.ylabel("Normalized Flux")
            plt.title(f"Theoretical Light Curve")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_file, dpi=300) 
            if show:
                plt.show()
            print(f"Light curve saved to {output_file}")
        
