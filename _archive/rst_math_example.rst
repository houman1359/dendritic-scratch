Example of usinng references. For example, :cite:p:`zhang2023gpu` is an example of p citation and :cite:t:`zhang2023gpu` is an example of t citation.

Here is a mathematical equation, The membrane potential dynamics in the Hodgkin-Huxley model are described by the following equation:

.. math::
    :nowrap:

    \begin{gather*}
    C_m \frac{dV}{dt} = I_{ext} - \left[ g_K n^4 (V - V_K) + g_{Na} m^3 h (V - V_{Na}) + g_L (V - V_L) \right]
    \end{gather*}

where: 

- :math:`C_m` is the membrane capacitance,
- :math:`V` is the membrane potential,
- :math:`I_{ext}` is the external applied current,
- :math:`g_K`, :math:`g_{Na}`, and :math:`g_L` are the conductances of potassium, sodium, and leak channels, respectively,
- :math:`V_K`, :math:`V_{Na}`, and :math:`V_L` are the reversal potentials for potassium, sodium, and leak channels,
- :math:`n`, :math:`m`, and :math:`h` are gating variables for potassium and sodium channels.