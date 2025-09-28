import numpy as np

def model_corrections(is_turbofan=True):
    models = {}

    # Correções de rotação e bypass
    models['N1_from_N2'] = np.poly1d([1.41661, -4.0478e-1])  # N1/N1_design = f(N2/N2_design)
    models['B_from_N1'] = np.poly1d([-8.3241e-1, 3.8824e-1, 1.4263])  # BPR/BPR_design = f(N1/N1_design)

    # Coeficientes para Prf
    models['A_from_B'] = np.poly1d([-0.00179, 0.00687, 0.5])  # A = f(B/B_design)
    models['C_from_B'] = np.poly1d([0.011, 0.53782])  # C = f(B/B_design)

    # Correções de pressão
    models['Pr_bst_from_N1'] = np.poly1d([4.8967e-1, -4.3317e-2, 5.6846e-1])  # Prc/Prc_design = f(N2/N2_design)
    models['Prc_from_N2'] = np.poly1d([-6.0730, 1.4821e1, -1.0042e1, 2.2915])  # Prc/Prc_design = f(N2/N2_design)

    # Correções de temperatura
    models['T04_from_N2'] = np.poly1d([8.1821e-1, -2.2401e-1, 4.1842e-1])  # T04/T04_design = f(N2/N2_design)

    # Correções de eficiência
    models['eta_f_from_N1'] = np.poly1d(
        [-6.6663, 17.752, -17.469, 7.7181, -0.32985])  # ηf/ηf_design = f(N1/N1_design)
    models['eta_c_from_N2'] = np.poly1d([-1.1234, 2.1097, 0.018617])  # ηc/ηc_design = f(N2/N2_design)
    models['eta_tf_from_N1'] = np.poly1d([-6.7490e-2, 0.25640, 0.81153])  # ηtf/ηtf_design = f(N1/N1_design)
    models['eta_t_from_N2'] = np.poly1d([-6.7490e-2, 0.25640, 0.81153])  # ηt/ηt_design = f(N2/N2_design)
    models['eta_b_from_N2'] = np.poly1d([1.1630, -3.0851, 2.7312, 0.19130])  # ηb/ηb_design = f(N2/N2_design)

    # Correção de vazão mássica
    models['m_dot_H_from_N2'] = np.poly1d([-6.6970, 1.7001e1, -1.2170e1, 2.8717])  # ṁ_H/ṁ_H_design = f(N2/N2_design)

    if is_turbofan:
        return models
    else:
        # Correções específicas para turboprop
        models['N1_from_N2'] = np.poly1d([0.0, 1.0])
        models['eta_turbina_livre_from_N2'] = np.poly1d([
            1.9062e1, -5.2456e1, 4.7887e1, -1.3489e1
        ])
        models['pr_tl_from_N2'] = np.poly1d([
            -1.8063e1, 4.2469e1, -3.1480e1, 8.0681
        ])
        return models
