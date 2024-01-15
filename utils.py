import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import scipy as sp

t = sym.symbols('t')

def R(theta, phi, psi):
    # ZXZ
    M=[
        [sym.cos(theta)*sym.cos(psi)-sym.cos(phi)*sym.sin(theta)*sym.sin(psi), -sym.cos(theta)*sym.sin(psi)-sym.cos(phi)*sym.cos(psi)*sym.sin(theta), sym.sin(theta)*sym.sin(phi)],
        [sym.cos(psi)*sym.sin(theta)+sym.cos(theta)*sym.cos(phi)*sym.sin(psi), sym.cos(theta)*sym.cos(phi)*sym.cos(psi)-sym.sin(theta)*sym.sin(psi), -sym.cos(theta)*sym.sin(phi)],
        [sym.sin(phi)*sym.sin(psi), sym.cos(psi)*sym.sin(phi), sym.cos(phi)],
    ]
    return sym.Matrix(M)

def Rx(x):
    return sym.Matrix([
        [1,0,0],
        [0,sym.cos(x),-sym.sin(x)],
        [0,sym.sin(x),sym.cos(x)],
    ])

def V(k, R1,R2):
    # return k*sym.trace((R1.T*R2)* R1.T * R2 *(R1.T*R2))
    return k*sym.trace((R1.T*R2)*(R1.T*R2))

def K(k, R1):
    return k*sym.trace((R1.T * R1.diff(t))*(R1.T * R1.diff(t)))

def Anis(k, R1, eq, dphi):
    # Re = Rx(dphi)*R(eq[0],eq[1],eq[2])*Rx(-dphi)
    Re = R(0,dphi,0)*R(eq[0],eq[1],eq[2])*R(0,-dphi,0)
    return k*sym.trace((R1.T * Re)*(R1.T * Re))

def coefficients(expresions, variables):
    """returns a [N,N] python list"""
    eqs = []
    vars = []
    for i in expresions:
        for j in i:
            eqs.append(j)

    for i in variables:
        for j in i:
            vars.append(j)

    N = len(vars)
    out = []
    for i in range(N):
        l = []
        for j in range(N):
            l.append((eqs[i].lhs).coeff(vars[j]))
        out.append(l)
    return sym.Matrix(out)

class System:
    def __init__(self, Nbase, indexConnections, dirconnections, a=1,k_v=1, k_k=1, k_a=1, dphi=0, Js=1):
        self.Nbase = Nbase
        self.indexConnections=indexConnections
        self.directionConnections = dirconnections
        self.a=a
        self.k = [sym.symbols("k_x"), sym.symbols("k_y")]
        self.k_v = k_v
        self.k_k = k_k
        self.k_a = k_a
        self.dphi=dphi

        t = sym.symbols('t')
        self.auxvar=[
                sym.Function(r"\theta_a")(t),
                sym.Function(r"\phi_a")(t),
                sym.Function(r"\psi_a")(t),
        ]
        self.auxvar2 = [
            sym.Function(r"\theta_b")(t),
            sym.Function(r"\phi_b")(t),
            sym.Function(r"\psi_b")(t),
        ]
        self.auxeq = [
            sym.symbols(r"\alpha"),
            sym.symbols(r"\beta"),
            sym.symbols(r"\gamma"),
        ]
        self.auxeq2 = [
            sym.symbols(r"\alpha_2"),
            sym.symbols(r"\beta_2"),
            sym.symbols(r"\gamma_2"),
        ]

        self.eigenvecs=None
        self.eigenvals=None

        if type(Js)==int: self.Js = 0 * self.indexConnections + self.k_v # todos tienen el mismo J 
        else:
            self.Js = Js  # debe ser un arreglo con el J de cada conexion


    def setEquilibrium(self, eq):
        t = sym.symbols('t')

        self.equilibrium = eq
        self.variables = []
        for i in range(self.Nbase):
            aux = [
                sym.Function(r"\theta_{}".format(i))(t),
                sym.Function(r"\phi_{}".format(i))(t),
                sym.Function(r"\psi_{}".format(i))(t),
            ]
            self.variables.append(aux)

        self.Rs = []
        for i in range(self.Nbase):
            R1 = R(
                self.equilibrium[i][0]+self.variables[i][0],
                self.equilibrium[i][1]+self.variables[i][1],
                self.equilibrium[i][2]+self.variables[i][2],
            )
            self.Rs.append(R1)

    def makeLagrangian(self):
        t = sym.symbols('t')

        Ls=[]

        for i in range(self.Nbase):
            kinetic = self.Kinetic(self.variables[i], self.equilibrium[i])
            potential = 0
            potential += self.Anisotropy(self.variables[i], self.equilibrium[i], i)
            for j in range(len(self.indexConnections[i])):
                idxconn = self.indexConnections[i][j]
                dirconn = self.directionConnections[i][j]
                k = 0
                for m in range(2):
                    k += dirconn[m]*self.k[m]
                factor = sym.exp(sym.I * self.a * k)
                if i==idxconn:
                    var = []
                    for z in self.auxvar:
                        var.append(z*factor)
                    potential += self.Potential(self.variables[i],self.equilibrium[i], var, self.equilibrium[idxconn], self.Js[i][j])
                else:
                    var=[]
                    for z in self.variables[idxconn]:
                        var.append(z*factor)
                    potential += self.Potential(self.variables[i], self.equilibrium[i], var,self.equilibrium[idxconn], self.Js[i][j])
            L = kinetic+potential
            Ls.append(L)

        self.lagrangians=Ls

    def equations(self):
        t = sym.symbols('t')

        self.E_L_equations=[]
        for i in range(self.Nbase):
            eq = sym.euler_equations(self.lagrangians[i], self.variables[i],t)
            self.E_L_equations.append(eq)

        # Replace varsaux por la variable correspondiente por el factor
        # esto solo es necesario si tenemos conexion a-a o b-b, etc
        
    def Kmatrix(self):
        t = sym.symbols('t')
        ddvar = []
        for i in self.variables:
            l = []
            for j in i:
                l.append(j.diff(t,2))
            ddvar.append(l)

        self.K = coefficients(self.E_L_equations, self.variables)
        self.W = -1*coefficients(self.E_L_equations, ddvar) # el -1 es por dt^2 -> -w^2

    def Wmatrix(self):
        newvar = []
        for i in range(self.Nbase):
            newvar.append(
                [
                    sym.symbols("a_{}".format(i)),
                    sym.symbols("b_{}".format(i)),
                    sym.symbols("c_{}".format(i)),
                ]
            )

        eqs = []
        for leq in self.E_L_equations:
            l = []
            for eq in leq:
                x = eq
                for i in range(self.Nbase):
                    for j in range(3):
                        ddvar = self.variables[i][j].diff(t, 2)
                        x = x.subs(ddvar, newvar[i][j])
                l.append(x)
            eqs.append(l)

        self.W = -1*coefficients(eqs, newvar)

    def seriesExpansion(self):
        epsilon = sym.symbols(r"\epsilon")
        littleauxvar1 = [
            self.auxeq[0] + epsilon*self.auxvar[0],
            self.auxeq[1] + epsilon*self.auxvar[1],
            self.auxeq[2] + epsilon*self.auxvar[2],
        ]
        littleauxvar2 = [
            self.auxeq2[0] + epsilon*self.auxvar2[0],
            self.auxeq2[1] + epsilon*self.auxvar2[1],
            self.auxeq2[2] + epsilon*self.auxvar2[2],
        ]
        R1 = R(littleauxvar1[0], littleauxvar1[1], littleauxvar1[2])
        R2 = R(littleauxvar2[0], littleauxvar2[1], littleauxvar2[2])

        # self.V = sym.series(V(self.k_v, R1, R2),epsilon, 0, 3).removeO().subs(epsilon,1)
        # self.K = sym.series(K(self.k_k, R1),epsilon, 0, 3).removeO().subs(epsilon,1)
        self.V = V(self.k_v, R1, R2).diff(epsilon, 2).subs(epsilon,0)
        self.K = K(self.k_k, R1).diff(epsilon, 2).subs(epsilon,0)
        self.A = Anis(self.k_a, R1, self.auxeq, self.auxeq2[0]).diff(epsilon, 2).subs(epsilon,0)
        
    def Potential(self, angles1,eq1, angles2, eq2, k):
        return self.V.subs([
            (self.auxvar[0], angles1[0]),(self.auxvar[1], angles1[1]),(self.auxvar[2], angles1[2]),
            (self.auxeq[0], eq1[0]),(self.auxeq[1], eq1[1]),(self.auxeq[2], eq1[2]),
            (self.auxvar2[0], angles2[0]),(self.auxvar2[1], angles2[1]),(self.auxvar2[2], angles2[2]),
            (self.auxeq2[0], eq2[0]),(self.auxeq2[1], eq2[1]),(self.auxeq2[2], eq2[2]),
            (self.k_v, k)
            ])

    def Kinetic(self, angles1, eq1):
        return self.K.subs([
            (self.auxvar[0], angles1[0]),(self.auxvar[1], angles1[1]),(self.auxvar[2], angles1[2]),
            (self.auxeq[0], eq1[0]),(self.auxeq[1], eq1[1]),(self.auxeq[2], eq1[2]),
            ])
    
    def Anisotropy(self, angles1, eq1, idx):
        return self.A.subs([
            (self.auxvar[0], angles1[0]),(self.auxvar[1], angles1[1]),(self.auxvar[2], angles1[2]),
            (self.auxeq[0], eq1[0]),(self.auxeq[1], eq1[1]),(self.auxeq[2], eq1[2]),
            (self.auxeq2[0], ((-1)**idx)*self.dphi),
        ])

    def eigen(self, N=50):
        ks = np.linspace(-np.pi/self.a, np.pi/self.a, N)
        self.eigenvals = np.zeros([N, self.Nbase*3])
        self.eigenvecs = np.zeros([N, self.Nbase*3, self.Nbase*3], dtype=complex)

        for i in range(N):
            A = np.array(self.K.subs(self.k[0], ks[i])).astype(dtype=complex)
            B = np.array(self.W.subs(self.k[0], ks[i])).astype(dtype=complex)
            eig = sp.linalg.eig(A,B)
            idx = np.argsort(eig[0].real)
            for j in range(self.Nbase*3):    
                self.eigenvals[i][j] = eig[0][idx[j]]
                # self.eigenvecs[i][j] = eig[1][idx[j]]
                vec = eig[1][idx[j]]
                if vec[0] != 0:
                    vec /= vec[0]
                vec /= np.linalg.norm(vec)
                self.eigenvecs[i][j] = vec # se elige el gauge tq el primer elemento es 1 (real)

    def eigen2D(self, N=50, print=0):
        kx = np.linspace(-np.pi/self.a, np.pi/self.a, N)
        ky = np.linspace(-np.pi/self.a, np.pi/self.a, N)
        X, Y = np.meshgrid(kx, ky)
        
        self.BZ = np.array([kx, ky])

        self.eigenvals = np.zeros([N,N,self.Nbase*3])
        self.eigenvecs = np.zeros([N,N,self.Nbase*3, self.Nbase*3], dtype=complex)

        for i in range(N):
            for j in range(N):
                A = np.array(self.K.subs([(self.k[0], X[i,j]), (self.k[1], Y[i,j])])).astype(dtype=complex)
                B = np.array(self.W.subs([(self.k[0], X[i,j]), (self.k[1], Y[i,j])])).astype(dtype=complex)
                eig = sp.linalg.eig(A,B)
                idx = np.argsort(eig[0].real)
                if print==1: print(eig[0])
                for k in range(self.Nbase*3):
                    self.eigenvals[i][j][k] = eig[0][idx[k]]
                    # self.eigenvecs[i][j][k] = eig[1][idx[k]]
                    vec = eig[1][idx[k]] / eig[1][idx[k]][0]
                    vec /= np.linalg.norm(vec)
                    self.eigenvecs[i][j][k] = vec

        fig = plt.figure()
        fig.clf()
        ax = plt.axes(projection='3d')
        # ax.plot_surface(X,Y, self.eigenvals[:,:,0])
        # ax.plot_surface(X,Y, self.eigenvals[:,:,1])
        ax.contour3D(X,Y,self.eigenvals[:,:,0],50, cmap='binary')
        ax.contour3D(X,Y,self.eigenvals[:,:,1],50,cmap='binary')



    def plotkx(self, N=50):
        ks = np.linspace(-np.pi/self.a, np.pi/self.a, N)
        y = np.zeros([N, self.Nbase*3])
        
        plt.plot(ks, self.eigenvals)

        
    def plot2Dband(self, N=50):
        Gamma = np.array([0,0])
        X = np.array([np.pi/self.a, 0])
        M = np.array([np.pi/self.a, np.pi/self.a])

        pathGX = np.outer(np.linspace(0, 1, N), X-Gamma) + Gamma
        pathXM = np.outer(np.linspace(0, 1, N), M-X) + X
        pathMG = np.outer(np.linspace(0, 1, N), Gamma-M) + M
        path = np.concatenate([pathGX, pathXM, pathMG])

        y = np.zeros([3*N, self.Nbase*3])

        for i in range(3*N):
            A = np.array(self.K.subs([(self.k[0], path[i][0]),(self.k[1], path[i][1])])).astype(dtype=complex)
            B = np.array(self.W.subs([(self.k[0], path[i][0]),(self.k[1], path[i][1])])).astype(dtype=complex)
            eig = sp.linalg.eig(A, B)[0]
            y[i] = np.sort(eig)
     
        
        k1 = np.linspace(0,1,N)
        k2 = np.linspace(1,2,N)
        k3 = np.linspace(2,np.pi,N)
        ks = np.concatenate([k1,k2,k3])
        plt.plot(ks,y)
        plt.axvline(1, color='black')
        plt.axvline(2, color='black')


    def plot2D(self, N=50, band1=0, band2=1):
        kx = np.linspace(-np.pi/self.a, np.pi/self.a, N)
        ky = np.linspace(-np.pi/self.a, np.pi/self.a, N)
        X, Y = np.meshgrid(kx, ky)
        
        fig = plt.figure()
        fig.clf()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X,Y,self.eigenvals[:,:,band1])
        ax.plot_surface(X,Y,self.eigenvals[:,:,band2])
        # ax.contour3D(X,Y,Z1,50, cmap='binary')
        # ax.contour3D(X,Y,Z2,50,cmap='binary')

    def initialize(self, equilibrium):
        self.setEquilibrium(equilibrium)
        self.seriesExpansion()
        self.makeLagrangian()
        self.equations()
        self.Kmatrix()
    
    def berryPhase(self,band=0):
        N = len(self.eigenvals[:,0])
        bp=0
        for i in range(N):
            # bpi = np.imag(np.dot(np.conj(self.eigenvecs[i][band])/np.linalg.norm(self.eigenvecs[i][band]),
                        #   self.eigenvecs[(i+1)%N][band]/np.linalg.norm(self.eigenvecs[(i+1)%N][band])-self.eigenvecs[i][0]/np.linalg.norm(self.eigenvecs[i][band]))
            # )
            bpi = np.imag(
                np.log(np.dot(
                    np.conj(self.eigenvecs[i][band]),
                    self.eigenvecs[(i+1)%N][band]
                            ) / np.linalg.norm(np.dot(
                    np.conj(self.eigenvecs[i][band]),
                    self.eigenvecs[(i+1)%N][band]
                            ))
                    )
            )
            print(bpi)
            bp += bpi
            # bpi = np.imag(
            #     np.log(np.dot(
            #         np.conj(self.eigenvecs[i][band])/np.linalg.norm(self.eigenvecs[i][band]),
            #         self.eigenvecs[(i+1)%N][band]/np.linalg.norm(self.eigenvecs[(i+1)%N][band])
            #                 )
            #         )
            # )
        self.bp=np.mod(bp, 2*np.pi)
        return self.bp
    
    def computeChern(self):
        BZ = self.BZ

        N = len(BZ[0])

        F = np.zeros([N,N,self.Nbase*3])
        U1 = np.zeros([N,N,self.Nbase*3], dtype=complex)
        U2 = np.zeros([N,N,self.Nbase*3], dtype=complex)

        for i in range(N):
            for j in range(N):
                for band in range(self.Nbase*3):
                    U1[i][j][band] = np.dot(
                        np.conj(self.eigenvecs[i][j][band]), self.eigenvecs[(i+1)%N][j][band]
                    )
                    U1[i][j][band] /= np.linalg.norm(U1[i][j][band])
                    U2[i][j][band] = np.dot(
                        np.conj(self.eigenvecs[i][j][band]), self.eigenvecs[i][(j+1)%N][band]
                    )
                    U2[i][j][band] /= np.linalg.norm(U2[i][j][band])
                    
        for i in range(N):
            for j in range(N):
                for band in range(self.Nbase*3):
                    F[i][j][band] = np.imag(np.log(
                        U1[i][j][band] * U2[(i+1)%N][j][band] * (U1[i][(j+1)%N][band])**-1 * (U2[i][j][band])**-1
                    ))
                    F[i][j][band] = np.mod(F[i][j][band], 2*np.pi)

        self.chern = np.zeros(self.Nbase*3)
        for band in range(self.Nbase*3):
            self.chern[band] = np.sum(F[:,:,band]) / (2*np.pi) 

        return self.chern
    
    
    def computeNonAbelianChern(self):
        BZ = self.BZ

        N = len(BZ[0])

        F = np.zeros([N,N])
        U1 = np.zeros([N,N], dtype=complex)
        U2 = np.zeros([N,N], dtype=complex)

        for i in range(N):
            for j in range(N):
                psiDag_psi1 = np.zeros([self.Nbase*3, self.Nbase*3])
                psiDag_psi2 = np.zeros([self.Nbase*3, self.Nbase*3])

                for band1 in range(self.Nbase*3):
                    for band2 in range(self.Nbase*3):

                        psiDag_psi1[band1][band2] = np.dot(
                            np.conj(self.eigenvecs[i][j][band1]), self.eigenvecs[(i+1)%N][j][band2]
                        ) 
                        psiDag_psi2[band1][band2] = np.dot(
                            np.conj(self.eigenvecs[i][j][band1]), self.eigenvecs[i][(j+1)%N][band2]
                        )
                det1 = np.linalg.det(psiDag_psi1)
                U1[i][j] = det1 / np.linalg.norm(det1)
                det2 = np.linalg.det(psiDag_psi2)
                U2[i][j] = det2 / np.linalg.norm(det2)

                    
        for i in range(N):
            for j in range(N):
                F[i][j] = np.imag(np.log(
                    U1[i][j] * U2[(i+1)%N][j] * (U1[i][(j+1)%N])**-1 * (U2[i][j])**-1
                ))
                F[i][j] = np.mod(F[i][j], 2*np.pi)

        self.chern = np.sum(F[:,:]) / (2*np.pi) 

        return self.chern

