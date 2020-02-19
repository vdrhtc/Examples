from single_transmon.Transmon import *
from qutip import *
from itertools import product

class TransmonChain:

    def __init__(self, length, transmon_truncation=2,
                 transmon_diagonalization_dimension=15):
        self._length = length
        self._transmon_truncation = transmon_truncation
        self._transmon_diagonalization_dimension = transmon_diagonalization_dimension
        self._Ec = []
        self._Ej = []
        self._J = []
        self._d = []
        self._Omega = []
        self._omega = None
        self._gamma_phi = []
        self._gamma_rel = []
        self._phi = []
        self._c_ops = None
        self._transmons = [None] * self._length
        self._zero_op = tensor(*[qzero(self._transmon_truncation) for i in range(0, self._length)])
        self._identity_array = [qeye(self._transmon_truncation) for i in range(0, self._length)]

        self._RWA_driving_cache = {}
        self._RWA_subtrahend_cache = {}
        self._transmon_H_cache = [{} for _ in range(self._length)]
        self._interaction_cache = {}

        self._low_energy_states = []
        self._low_energy_state_indices = []

    def clear_caches(self):
        self._RWA_driving_cache = {}
        self._RWA_subtrahend_cache = {}
        self._transmon_H_cache = [{} for _ in range(self._length)]
        self._interaction_cache = {}

    def build_low_energy_kets(self, total_population):
        self._low_energy_states = []
        self._low_energy_state_indices = []
        transmon_states = [list(range(self._transmon_truncation)) for i in range(self._length)]
        for idx, state_combination in enumerate(product(*transmon_states)):
            if sum(state_combination) <= total_population:
                self._low_energy_states.append(state_combination)
                self._low_energy_state_indices.append(idx)
        print("Total %d kets included" % len(self._low_energy_states))

    def build_H_RWA(self):
        return self.build_H_full() - self.build_RF_subtrahend()

    def build_H_full(self):
        H_chain = self._zero_op.copy()

        for i in range(0, self._length):
            H_chain += self._build_transmon_H_at_index(i)
        for i in range(0, self._length - 1):
            H_chain += self._build_interaction_RWA(i, i + 1)
        return H_chain

    def build_c_ops(self):  # building at 0 flux since no real dependence on anything
        if self._c_ops is not None:
            return self._c_ops

        self._c_ops = []

        for i in range(self._length):
            chain_operator_rel = self._identity_array.copy()
            chain_operator_rel[i] = self._transmons[i].c_ops(0)[0]
            chain_operator_deph = self._identity_array.copy()
            chain_operator_deph[i] = self._transmons[i].c_ops(0)[1]
            self._c_ops.append(tensor(*chain_operator_rel))
            self._c_ops.append(tensor(*chain_operator_deph))
        return self._c_ops

    def truncate_to_low_population_subspace(self, operator):
        new_oper = []
        for idx1 in self._low_energy_state_indices:
            new_oper_row = []
            for idx2 in self._low_energy_state_indices:
                new_oper_row.append(operator[idx1, idx2])
            new_oper.append(new_oper_row)
        return Qobj(new_oper)

    def build_RWA_driving(self):
        try:
            return self._RWA_driving_cache[tuple(self._Omega)]
        except:
            driving_operator = self._zero_op.copy()
            for i in range(0, self._length):
                chain_operator = self._identity_array.copy()
                chain_operator[i] = self._transmons[i].Hdr_cont_RF_RWA(self._Omega[i])
                driving_operator += tensor(*chain_operator)
            self._RWA_driving_cache[tuple(self._Omega)] = driving_operator
            return driving_operator

    def build_RF_subtrahend(self):
        try:
            return self._RWA_subtrahend_cache[self._omega]
        except:
            subtrahend = self._zero_op.copy()
            for i in range(0, self._length):
                chain_operator = self._identity_array.copy()
                chain_operator[i] = self._omega * ket2dm(self._transmons[i].e_state()) + \
                                    self._omega * 2 * ket2dm(
                    self._transmons[
                        i].f_state()) if self._transmon_truncation is 3 else self._omega * ket2dm(
                    self._transmons[i].e_state())
                subtrahend += tensor(*chain_operator)
            self._RWA_subtrahend_cache[self._omega] = subtrahend
            return subtrahend

    def _build_transmon_H_at_index(self, i):
        try:
            return self._transmon_H_cache[i][self._phi[i]]
        except:
            chain_operator = self._identity_array.copy()
            chain_operator[i] = self._transmons[i].H_diag_trunc(self._phi[i])
            self._transmon_H_cache[i][self._phi[i]] = tensor(*chain_operator)
            return self._transmon_H_cache[i][self._phi[i]]

    def _build_interaction_RWA(self, i, j):
        try:
            return self._interaction_cache[(i, j)][(self._phi[i], self._phi[j])]
        except:
            chain_operator1 = self._identity_array.copy()
            chain_operator1[i] = self._transmons[i].raising(self._phi[i])
            chain_operator1[j] = self._transmons[j].lowering(self._phi[j])

            chain_operator2 = self._identity_array.copy()
            chain_operator2[i] = self._transmons[i].lowering(self._phi[i])
            chain_operator2[j] = self._transmons[j].raising(self._phi[j])

            if (i,j) not in self._interaction_cache:
                self._interaction_cache[(i,j)] = {}
            self._interaction_cache[(i,j)][(self._phi[i], self._phi[j])] = \
                self._J[i] * (tensor(*chain_operator1) + tensor(*chain_operator2))
            return self._interaction_cache[(i,j)][(self._phi[i], self._phi[j])]

    def build_transmons(self):
        for i in range(self._length):
            self._transmons[i] = Transmon(self._Ec[i], self._Ej[i], self._d[i],
                                          self._gamma_rel[i], self._gamma_phi[i],
                                          self._transmon_diagonalization_dimension,
                                          self._transmon_truncation, i)

    def get_length(self):
        return self._length

    def set_Ec(self, Ec):
        try:
            assert len(Ec) == self._length
            self._Ec = Ec
        except TypeError:
            self._Ec = [Ec] * self._length
            print("Setting all Ec to be equal")
        except AssertionError:
            raise ValueError("Length of Ec is not equal to the number of transmons")

    def set_Ej(self, Ej):
        try:
            assert len(Ej) == self._length
            self._Ej = Ej
        except TypeError:
            self._Ej = [Ej] * self._length
            print("Setting all Ej to be equal")
        except AssertionError:
            raise ValueError("Length of Ej is not equal to the number of transmons")

    def set_J(self, J):
        try:
            assert len(J) == self._length
            self._J = J
        except TypeError:
            self._J = [J] * self._length
            print("Setting all J to be equal")
        except AssertionError:
            raise ValueError("Length of J is not equal to the number of transmons")

    def set_Omega(self, Omega):
        try:
            assert len(Omega) == self._length
            self._Omega = Omega
        except TypeError:
            self._Omega = [Omega] * self._length
            print("Setting all Omega to be equal")
        except AssertionError:
            raise ValueError("Length of Omega is not equal to the number of transmons")

    def set_omega(self, omega):
        self._omega = omega

    def set_asymmetry(self, d):
        try:
            assert len(d) == self._length
            self._d = d
        except TypeError:
            self._d = [d] * self._length
            print("Setting all d to be equal")
        except AssertionError:
            raise ValueError("Length of d is not equal to the number of transmons")

    def set_gamma_rel(self, gamma_rel):
        try:
            assert len(gamma_rel) == self._length
            self._gamma_rel = gamma_rel
        except TypeError:
            self._gamma_rel = [gamma_rel] * self._length
            print("Setting all gamma_rel to be equal")
        except AssertionError:
            raise ValueError("Length of gamma_rel is not equal to the number of transmons")

    def set_gamma_phi(self, gamma_phi):
        try:
            assert len(gamma_phi) == self._length
            self._gamma_phi = gamma_phi
        except TypeError:
            self._gamma_phi = [gamma_phi] * self._length
            print("Setting all gamma_phi to be equal")
        except AssertionError:
            raise ValueError("Length of gamma_phi is not equal to the number of transmons")

    def set_phi(self, phi):
        try:
            assert len(phi) == self._length
            self._phi = phi
        except TypeError:
            self._phi = [phi] * self._length
            print("Setting all fluxes to be equal")
        except AssertionError:
            raise ValueError(
                "Length of fluxes is not equal to the number of transmons: " + str(phi))
