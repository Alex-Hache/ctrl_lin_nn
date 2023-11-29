import matlab.engine

def extract_parameter_name(module_name):
    # Divise la chaîne en parties en utilisant le caractère '.'
    parts = module_name.split('.')

    # Vérifie la présence de 'linmod'
    if 'linmod' in parts and not 'original' in parts:
        return parts[-1]

    # Vérifie la présence de 'original'
    if 'original' in parts:
        return parts[-2]

    # Vérifie la présence de 'weight' ou 'bias'
    if 'weight' in parts or 'bias' in parts:
        if 'bias' in parts:
            return parts[-2] + '_bias'
        else:
            return parts[-2]

    # Retourne None si aucune correspondance n'est trouvée
    return parts[-1]

def sim_closed_loop(strMatFileWeights, net_dims, dt, strNameSaveFig : str ):
    dt = matlab.double([dt])
    eng = matlab.engine.start_matlab()
    net_dims = matlab.double(net_dims)
    eng.addpath(eng.genpath(eng.pwd()))

    eng.load_workspace(strMatFileWeights, net_dims, dt, strNameSaveFig, nargout=0)
    eng.closedLoopresults_pendulum(nargout=0)
    eng.quit()