from dolfinx import fem as fe
from dolfinx import mesh,io
from mpi4py import MPI
import numpy as np
import ufl
from ufl import (dot,div, as_tensor, as_vector, inner, dx, Measure, sqrt,conditional)
from petsc4py.PETSc import ScalarType
from boundarycondition import BoundaryCondition,MarkBoundary
from dataclasses import dataclass
from constants import g, R, omega, p_water, p_air
from forcing import GriddedForcing


@dataclass
class BaseProblem:
    """Steady-state problem on a unit box
    """
    h_init: float = None
    nx: int = 10
    ny: int = 10
    #define separatley for mixed elements
    h_0: callable = lambda x: np.sin(x[0]*np.pi)*np.sin(x[1]*np.pi)+1.0
    v_0: callable  = lambda x: np.vstack([ np.ones(x.shape[1]),np.ones(x.shape[1])])
    TAU: float = 0.0
    h_b: float = 10.0
    solution_var: str = 'eta'
    friction_law: str = 'linear'
    spherical: bool = False
    # applied only if spherical is enabled
    projected: bool = True
    # path to forcing file
    forcing: GriddedForcing = None
    lat0: float = 35

    def __post_init__(self):
        """Initialize the mesh and other variables needed for BC's
        """
        
        self._create_mesh()
        self._dirichlet_bcs = None
        self._boundary_conditions = None

    def _create_mesh(self):
        self.mesh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
    
    def init_V(self, V):
        """Initialize the space V in which the problem will be solved.
        
        This is defined in the solver, which then calls this method on initialization.
        The first subspace of V MUST correspond to h.
        """

        self.V = V
        # initialize exact solution
        self.u_ex = fe.Function(V)

        #part of rewrite for mixed element
        self.u_ex.sub(0).interpolate(self.h_0)
        self.u_ex.sub(1).interpolate(self.v_0)
        scalar_V = V.sub(0).collapse()[0]
        if self.spherical:
            self.S = fe.Function(scalar_V)
            self.tan = fe.Function(scalar_V)
            self.sin = fe.Function(scalar_V)
            if self.projected:
                self.S.interpolate(lambda x: np.cos(np.deg2rad(self.lat0))/np.cos(x[1]/R))
                self.tan.interpolate(lambda x: np.tan(x[1]/R))                
                self.sin.interpolate(lambda x: np.sin(x[1]/R))                
            else:
                # raw spherical
                self.S.interpolate(lambda x: 1./np.cos(x[1]))
                self.tan.interpolate(lambda x: np.tan(x[1]))
                self.sin.interpolate(lambda x: np.sin(x[1]))


        if self.forcing is not None:
            self.forcing.set_V(scalar_V)
            self.forcing.evaluate(self.t)

    def _get_standard_vars(self, u, form='h'):
        """Return a standardized representation of the solution variables
        """

        if self.solution_var == 'h':
            h, ux, uy = u[0], u[1], u[2]
            eta = h - self.h_b
            hux, huy = h*ux, h*uy
        elif self.solution_var == 'eta':
            eta, ux, uy = u[0], u[1], u[2]
            h = eta + self.h_b
            hux, huy = h*ux, h*uy
        elif self.solution_var == 'flux':
            h, hux, huy = u[0], u[1], u[2]
            eta = h - self.h_b
            ux, uy = hux / h, huy / h
        else:
            raise ValueError(f"Invalid solution variable '{self.solution_var}'")

        if form == 'h': return h, ux, uy
        elif form == 'eta': return eta, ux, uy
        elif form == 'flux': return h, hux, huy
        else:
            raise ValueError(f"Invalid output form '{form}'")

    def make_Fu(self, u):
        h, ux, uy = self._get_standard_vars(u, form='h')
        #        components = [
        #            [h*ux,h*uy], 
        #            [h*ux*ux+ 0.5*g*h*h, h*ux*uy],
        #            [h*ux*uy,h*uy*uy+0.5*g*h*h]
        #        ]
        #well balanced from Kubatko paper    
        components = [
            [h*ux,h*uy], 
            [h*ux*ux+ 0.5*g*h*h-0.5*g*self.h_b*self.h_b, h*ux*uy],
            [h*ux*uy,h*uy*uy+0.5*g*h*h-0.5*g*self.h_b*self.h_b]
        ]    
        
        if self.spherical:
            # add spherical correction factor
            for i in range(len(components)):
                components[i][0] = components[i][0] * self.S
            if self.projected:
                return as_tensor(components)
            else:
                return as_tensor(components) / R
        else:
            return as_tensor(components)

    def make_Fu_wall(self, u):
        h, ux, uy = self._get_standard_vars(u, form='h')
        #        components = [
        #            [0,0], 
        #            [ 0.5*g*h*h, 0],
        #            [0,0.5*g*h*h ]
        #        ]
        #for well balanced
        components = [
            [0,0], 
            [0.5*g*h*h-0.5*g*self.h_b*self.h_b, 0],
            [0,0.5*g*h*h-0.5*g*self.h_b*self.h_b]
        ]

        if self.spherical:
            # add spherical correction factor
            #Mark messing with things
            for i in range(len(components)):
                components[i][0] = components[i][0] * self.S
            if self.projected:
                #just write our own
                #components = [
                #    [(self.S-1)*h*ux,0], 
                #    [ (self.S-1)*(h*ux*ux)+self.S*0.5*g*h*h, 0],
                #    [(self.S-1)*h*ux*uy,0.5*g*h*h ]
                #    ]
                return as_tensor(components)
            else:
                return as_tensor(components) / R
        else:
            return as_tensor(components)

    def make_Fu_linearized(self,u):
        '''
        routine for computing momentum flux for linearized swe
        as used for manufactured solution test cases
        '''
        h, ux, uy = self._get_standard_vars(u, form='h')
        components = [
            [h*ux,h*uy], 
            [g*h-g*self.h_b, 0.0],
            [0.0,g*h-g*self.h_b]
        ]    
        
        if self.spherical:
            # add spherical correction factor
            for i in range(len(components)):
                components[i][0] = components[i][0] * self.S
            if self.projected:
                return as_tensor(components)
            else:
                return as_tensor(components) / R
        else:
            return as_tensor(components)

    def make_Fu_wall_linearized(self, u):
        h, ux, uy = self._get_standard_vars(u, form='h')
        #        components = [
        #            [0,0], 
        #            [ 0.5*g*h*h, 0],
        #            [0,0.5*g*h*h ]
        #        ]
        #for well balanced
        components = [
            [0,0], 
            [g*h-g*self.h_b, 0],
            [0,g*h-g*self.h_b]
        ]

        if self.spherical:
            # add spherical correction factor
            #Mark messing with things
            for i in range(len(components)):
                components[i][0] = components[i][0] * self.S
            if self.projected:
                #just write our own
                #components = [
                #    [(self.S-1)*h*ux,0], 
                #    [ (self.S-1)*(h*ux*ux)+self.S*0.5*g*h*h, 0],
                #    [(self.S-1)*h*ux*uy,0.5*g*h*h ]
                #    ]
                return as_tensor(components)
            else:
                return as_tensor(components) / R
        else:
            return as_tensor(components)


    def get_friction(self, u):
        friction_law = self.friction_law
        h, ux, uy = self._get_standard_vars(u, form='h')
        if friction_law == 'linear':
            cf = self.TAU
            #linear law which is same as ADCIRC option
            return as_vector((0,
                 ux*cf,
                uy*cf))
        elif friction_law == 'quadratic':
            #experimental but 1e-16 seems to be ok
            eps = 1e-8
            self.TAU = 0.003
            vel_mag = conditional(pow(ux*ux + uy*uy, 0.5) < eps, eps, pow(ux*ux + uy*uy, 0.5))
            return as_vector(
                (0,
                ux*vel_mag*self.TAU,
                uy*vel_mag*self.TAU ) )

        elif friction_law == 'mannings':
            #experimental but 1e-16 seems to be ok
            eps = 1e-8
            mag_v = conditional(pow(ux*ux + uy*uy, 0.5) < eps, eps, pow(ux*ux + uy*uy, 0.5))
            return as_vector(
                (0,
                g*self.TAU_const*self.TAU_const*ux*mag_v*pow(h,-1/3),
                g*self.TAU_const*self.TAU_const*uy*mag_v*pow(h,-1/3)) )
        
        elif friction_law == 'nolibf2':
            eps=1e-8
            mag_v = conditional(pow(ux*ux + uy*uy, 0.5) < eps, eps, pow(ux*ux + uy*uy, 0.5))
            FFACTOR = 0.0025
            HBREAK = 1.0
            FTHETA = 10.0
            FGAMMA = 1.0/3.0
            Cd = conditional(h>eps, (FFACTOR*(1+HBREAK/h)**FTHETA)**(FGAMMA/FTHETA), eps  )
            return as_vector(
                (0,
                Cd*ux*mag_v,
                Cd*uy*mag_v) )            

    
    def make_Source(self, u,form='well_balanced'):
        h, ux, uy = self._get_standard_vars(u, form='h')
        if self.spherical:
            if self.projected:
                #canonical form is necessary for SUPG terms
                if form != 'well_balanced':
                    g_vec = as_vector(
                        (
                            -h * uy * self.tan / R,
                            -g*h*self.h_b.dx(0) * self.S - h * ux * uy * self.tan / R - 2*uy*omega*self.sin,
                            -g*h*self.h_b.dx(1) + h * ux * ux * self.tan / R + 2*ux*omega*self.sin
                        )
                    )
                #well balanced is default
                else:
                    g_vec = as_vector(
                        (
                            -h * uy * self.tan / R,
                            -g*(h-self.h_b)*self.h_b.dx(0) * self.S - h * ux * uy * self.tan / R - 2*uy*omega*self.sin,
                            -g*(h-self.h_b)*self.h_b.dx(1) + h * ux * ux * self.tan / R + 2*ux*omega*self.sin
                        )
                    )
            else:
                if form != 'well_balanced':
                    g_vec = as_vector(
                        (
                            -h * uy * self.tan / R,
                            -g*h*self.h_b.dx(0) * self.S / R - h * ux * uy * self.tan / R - 2*uy*h*omega*self.sin,
                            -g*h*self.h_b.dx(1) / R + h * ux * ux * self.tan / R + 2*ux*h*omega*self.sin
                        )
                    )
                #well balanced
                else:
                    g_vec = as_vector(
                        (
                            -h * uy * self.tan / R,
                            -g*(h-self.h_b)*self.h_b.dx(0) * self.S / R - h * ux * uy * self.tan / R - 2*uy*omega*self.sin,
                            -g*(h-self.h_b)*self.h_b.dx(1) / R + h * ux * ux * self.tan / R + 2*ux*omega*self.sin
                        )
                    )
        else:
            if form != 'well_balanced':
                g_vec = as_vector(
                    (
                        0,
                        -g*h*self.h_b.dx(0),
                        -g*h*self.h_b.dx(1)
                    )
                )

            #well balanced is default
            else:
                g_vec = as_vector(
                    (
                        0,
                        -g*(h-self.h_b)*self.h_b.dx(0),
                        -g*(h-self.h_b)*self.h_b.dx(1)
                    )
                )


        source = g_vec + self.get_friction(u) 

        if self.forcing is not None:
            windx, windy, pressure = self.forcing.windx, self.forcing.windy, self.forcing.pressure
            wind_mag = pow(windx*windx + windy*windy, 0.5)
            drag_coeff = (0.75 + 0.067 * wind_mag) * 1e-3
            wind_forcing_terms = [
                0,
                -drag_coeff * (p_air / p_water) * windx * wind_mag,
                -drag_coeff * (p_air / p_water) * windy * wind_mag,
            ]
            #wind_forcing_terms = [0, 30*.001 * windx*wind_mag * (p_air/p_water), 30*.001 * windy *wind_mag * (p_air/p_water)] 

            #wind_vec = as_vector(wind_forcing_terms)
            #wind_form = dot(wind_vec, wind_vec) * dx
            #print("Initial wind forcing", fe.assemble_scalar(fe.form(wind_form))**.5)
            #raise ValueError()
            
            pressure_forcing_terms = [
                0,
                h * pressure.dx(0) / (p_water),
                h * pressure.dx(1) / (p_water)
            ]
            if self.spherical:
                pressure_forcing_terms[1] *= self.S
                if not self.projected:
                    pressure_forcing_terms[1] /= R
                    pressure_forcing_terms[2] /= R

            source += as_vector(wind_forcing_terms) + as_vector(pressure_forcing_terms)
            #source += as_vector(pressure_forcing_terms)

        return source

    def make_Source_linearized(self, u,form='well_balanced'):
        h, ux, uy = self._get_standard_vars(u, form='h')
        #just a linear friction term
        cf = 0.0001
        print("Linear source terms!! Using friction coefficient of ",cf)
        #linear law which is same as ADCIRC option
        return as_vector((0,
                 ux*cf,
                uy*cf))
       
    def init_bcs(self):
        """Create the boundary conditions
        """
        
        def open_boundary(x):
        	return np.isclose(x[0],0) | np.isclose(x[1],0)

        def closed_boundary(x):
        	return np.isclose(x[0],1) | np.isclose(x[1],1)

        # dealing with a vector u formulation, so adapt accordingly
        dofs_open = fe.locate_dofs_geometrical((self.V.sub(0), self.V.sub(0).collapse()[0]), open_boundary)[0]
        
        bcs = [fe.dirichletbc(self.u_ex.sub(0), dofs_open)]

        ux_dofs_closed = fe.locate_dofs_geometrical((self.V.sub(1), self.V.sub(1).collapse()[0]), closed_boundary)[0]
        uy_dofs_closed = fe.locate_dofs_geometrical((self.V.sub(2), self.V.sub(2).collapse()[0]), closed_boundary)[0]
        bcs += [fe.dirichletbc(self.u_ex.sub(1), ux_dofs_closed), fe.dirichletbc(self.u_ex.sub(2), uy_dofs_closed)]
        self._dirichlet_bcs = bcs

    def get_rhs(self):
        """Return the RHS (forcing term)
        """

        return div(self.make_Fu(self.u_ex))

    def l2_norm(self, vec):

        return (fe.assemble_scalar(fe.form(inner(vec, vec)*dx)))**.5

    def check_solution(self, u_sol):
        """Check the solution returned by a solver
        """


        parts = u_sol.split()
        eta_sol = parts[0]
        ux_sol, uy_sol = parts[1], parts[2]

        #evaluate error
        e0 = eta_sol-self.u_ex.sub(0)
        print('L2 error for eta:', self.l2_norm(e0))
        e1 = ux_sol-self.u_ex.sub(1)
        print('L2 error for u:', self.l2_norm(e1))
        e2 = uy_sol - self.u_ex.sub(2)
        print('L2 error for v', self.l2_norm(e2))

    def plot_solution(self,u_sol,filename,t=0):
        #takes a function and plots as 
        xdmf = io.XDMFFile(self.mesh.comm, filename+"/"+filename+".xdmf", "w")
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(u_sol,t)
        xdmf.close()

    @property
    def dirichlet_bcs(self):
        if self._dirichlet_bcs is None:
            self.init_bcs()

        return self._dirichlet_bcs

    @property
    def boundary_conditions(self):
        if self._boundary_conditions is None:
            self.init_bcs()

        return self._boundary_conditions
d