import numpy as np


def LIF1(Isyn):
    T = 150  # total time to simulate (msec)
    dt = 2  # simulation time step (msec)
    time = np.arange(0, T + dt, dt)  # time array
    t_rest = 0  # initial refectory time
    # LIF properties
    Vm = np.zeros((len(time), len(time)))  # potential (V) trace over time
    Rm = 1  # resistance (kOhm)
    Cm = 10  # capacitance (uF)
    tau_m = Rm * Cm  # time constant (msec)
    tau_ref = 4  # refractory period (msec)
    Vth = 1  # spike threshold (V)
    V_spike = 0.5  # spike delta (V)
    # Input stimulus
    I = 1.5  # input current (A)
    t = 0

    TT = np.array([])
    k = 0
    for i in range(len(time)):  # in enumerate(time)
        if t > t_rest:
            Vm[i + 1] = Vm[i] + (-Vm[i] + Isyn[i]*Rm) / tau_m * dt
            if Vm[i].all() >= Vth:
                Vm[i + 1] = Vm[i] + V_spike
                TT = np.append(TT, t)
                t_rest = t + tau_ref
                k += 1
        t += dt

    return Vm, TT


def main():
    input = np.arange(0, 152, 2)
    Vout, O = LIF1(input)
    print(Vout)
    print(O)


if __name__ == '__main__':
    main()
